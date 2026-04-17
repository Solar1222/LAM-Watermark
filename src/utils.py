import torch
import numpy as np
import librosa
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mel_from_wav(wav_path, target_frames=2048*8, hop_length=160, n_fft=1024):
    wav, sr = librosa.load(wav_path, sr=16000)
    required_samples = (target_frames - 1) * hop_length + n_fft
    current_samples = len(wav)
    
    if current_samples < required_samples:
        pad_amount = required_samples - current_samples
        wav = np.pad(wav, (0, pad_amount), mode='constant', constant_values=0)
    elif current_samples > required_samples:
        wav = wav[:required_samples]
        
    waveform = torch.from_numpy(wav).unsqueeze(0).to(device)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=n_fft, win_length=n_fft, hop_length=hop_length,
        f_min=0, f_max=8000, n_mels=64, center=False, 
        norm='slaney', mel_scale='slaney'
    ).to(device)
    
    mel = mel_transform(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mel = mel.transpose(1, 2).unsqueeze(1) 
    return mel.to(dtype=torch.float16)


def run_inversion(pipe, inv_scheduler, latents_start, encoder_hidden_states, encoder_hidden_states_1, vocab_indices, guidance_scale=6.0):
    """
    Perform DDIM Inversion to map latents back to noise.
    Moved to utils.py for modularity.
    """
    # Temporarily switch to inversion scheduler
    original_scheduler = pipe.scheduler
    pipe.scheduler = inv_scheduler
    
    latents = latents_start.clone()
    
    # Inversion process
    for t in pipe.scheduler.timesteps:
        with torch.no_grad():
            # Classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = pipe.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_1=encoder_hidden_states_1,
                vocab_indices=vocab_indices
            ).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step backward in the ODE (Inversion)
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
    # Restore original scheduler
    pipe.scheduler = original_scheduler
    return latents