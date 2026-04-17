import torch
import numpy as np
import os
import soundfile as sf
from tqdm import tqdm
from diffusers import AudioLDM2Pipeline, DDIMInverseScheduler, DDIMScheduler
from transformers import GPT2LMHeadModel

# Import custom modules from the src package
from src.attacks import AudioAttacks
from src.watermark import LAM
from src.utils import get_mel_from_wav

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_INFERENCE_STEPS = 50 
GUIDANCE_SCALE = 6.0

L_SHAPE = (1, 8, 512, 16) 
NUM_EXPERIMENTS = 5
SAMPLE_RATE = 16000
MARK_BPS = 100


ATTACK_TYPES = [
    ('clean', 'Original (No Attack)'),
    ('pink_noise_20db', 'Pink Noise (SNR=20dB)'),
    ('random_noise_20db', 'Gaussian Noise (SNR=20dB)'),
    ('lowpass_3k', 'Low-pass Filter (3kHz)'),
    ('bandpass_300_8k', 'Band-pass Filter (0.3-8kHz)'),
    ('stretch_2', 'Time Stretching (2x Speed)'),
    ('crop_10', 'Cropping (10% Front/Back)'),
    ('echo', 'Echo Effect'),
]

print(f"Initializing AudioLDM2 Pipeline on {DEVICE}...")

pipe = AudioLDM2Pipeline.from_pretrained(
    "cvssp/audioldm2", 
    torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32
).to(DEVICE)

# Optimization: Enable attention slicing for lower VRAM usage
pipe.enable_attention_slicing()

# Load the language model component specifically for prompt processing
pipe.language_model = GPT2LMHeadModel.from_pretrained(
    pipe.language_model.name_or_path
).to(DEVICE).half()

# Set up schedulers for generation and inversion
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
inv_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

tm_audio = LAM(
    latent_shape=L_SHAPE,
    mark_bps=MARK_BPS
    )
audio_attacks = AudioAttacks()

def get_encoded_prompt(prompt: str):
    """Encodes the text prompt into hidden states for the diffusion model."""
    with torch.no_grad():
        out = pipe.encode_prompt(
            prompt, 
            device=DEVICE, 
            do_classifier_free_guidance=True, 
            num_waveforms_per_prompt=1
        )
    return out

def run_inversion(latents_start, encoder_hidden_states, encoder_hidden_states_1, vocab_indices):
    """
    Performs DDIM Inversion to map a latent state back to noise.
    Used for embedding watermarks within the generation trajectory.
    """
    pipe.scheduler = inv_scheduler
    latents = latents_start.clone()
    
    for t in pipe.scheduler.timesteps:
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = pipe.unet(
                latent_model_input, t, 
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_1=encoder_hidden_states_1,
                vocab_indices=vocab_indices
            ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)
            
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return latents

def detect_watermark_from_audio(audio_path, tm_instance):
    """
    Loads an audio file, converts it to latent space via VAE, 
    and attempts to extract the watermark bit-string.
    """
    mel = get_mel_from_wav(audio_path, target_frames=tm_instance.latent_h * 4).to(DEVICE)

    with torch.no_grad():
        latents = pipe.vae.encode(mel).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

    acc, is_detected = tm_instance.eval_watermark(latents)
    return acc, is_detected


#Main Experiment Loop
def main():
    prompt = "A high quality recording of a peaceful forest with birds chirping"
    results = {at[0]: [] for at in ATTACK_TYPES}
    
    print(f"\nStarting Robustness Evaluation: {NUM_EXPERIMENTS} trials\n")

    enc_out = get_encoded_prompt(prompt)
    
    for i in range(NUM_EXPERIMENTS):
        print(f"--- Experiment {i+1}/{NUM_EXPERIMENTS} ---")
        
        # A. Create Watermarked Latent (Gaussian noise rearrangement)
        # Using a unique seed for each experiment
        init_latents, watermark_bits = tm_audio.create_watermark_and_return_w(seed=2026+i)
        init_latents = init_latents.to(DEVICE).half()
        
        # B. Generate Audio (Decoding the watermarked noise)
        with torch.no_grad():
            output_audio = pipe(
                prompt_embeds=enc_out[0],
                negative_prompt_embeds=enc_out[1],
                pooled_prompt_embeds=enc_out[2],
                negative_pooled_prompt_embeds=enc_out[3],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                latents=init_latents,
                output_type="np"
            ).audios[0]
        
        # C. Robustness Testing Loop
        for attack_id, attack_name in ATTACK_TYPES:
            temp_path = f"temp_{attack_id}.wav"
            
            # Apply selected attack
            if attack_id == 'clean':
                processed_audio = output_audio
            elif attack_id == 'pink_noise_20db':
                processed_audio = audio_attacks.pink_noise_attack(output_audio, snr_db=20.0)
            elif attack_id == 'random_noise_20db':
                processed_audio = audio_attacks.random_noise_attack(output_audio, snr_db=20.0)
            elif attack_id == 'lowpass_3k':
                processed_audio = audio_attacks.lowpass_filter_3k(output_audio)
            elif attack_id == 'bandpass_300_8k':
                processed_audio = audio_attacks.bandpass_filter_300_8k(output_audio)
            elif attack_id == 'stretch_2':
                processed_audio = audio_attacks.stretch_2(output_audio)
            elif attack_id == 'crop_10':
                processed_audio = audio_attacks.cropping_front_back(output_audio, crop_ratio=0.1)
            elif attack_id == 'echo':
                processed_audio = audio_attacks.echo_default(output_audio)
            
            # Save and Reload for detection (simulates real-world transmission)
            sf.write(temp_path, processed_audio, SAMPLE_RATE)
            acc, detected = detect_watermark_from_audio(temp_path, tm_audio)
            results[attack_id].append(acc)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        print(f"Trial {i+1} complete. Current Avg Acc: {np.mean(results['clean']):.4f}")


    #Final Report Generation
    print("\n" + "="*50)
    print(f"{'Attack Type':<30} | {'Mean Bit Accuracy':<15}")
    print("-" * 50)
    
    for attack_id, attack_name in ATTACK_TYPES:
        mean_acc = np.mean(results[attack_id])
        print(f"{attack_name:<30} | {mean_acc:<15.4f}")
    
    print("="*50)
if __name__ == "__main__":
    main()