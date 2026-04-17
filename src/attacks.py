import torch
import numpy as np
import typing as tp
import julius

# Automatic device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_pink_noise(length: int) -> torch.Tensor:
    """
    Generate pink noise using the Voss-McCartney algorithm with PyTorch.
    Pink noise has equal power per octave, making it more 'natural' than white noise.
    """
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
    
    # Normalization to peak amplitude
    pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
    return pink_noise

class AudioAttacks:
    """
    A comprehensive suite of audio attacks for evaluating watermark robustness.
    Includes noise injection, filtering, temporal stretching, and spatial effects.
    """
    
    @staticmethod
    def pink_noise_attack(
        audio_np: np.ndarray,
        snr_db: float = 20.0,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Adds pink noise to the audio with a controlled Signal-to-Noise Ratio (SNR).
        """
        audio_tensor = torch.from_numpy(audio_np).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, length]
        elif audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)  # [1, channels, length]
        
        audio_tensor = audio_tensor.to(device)
        
        # Calculate signal power
        signal_power = torch.mean(audio_tensor ** 2)
        
        # Generate pink noise and move to device
        pink_noise = generate_pink_noise(audio_tensor.shape[-1]).to(device)
        
        # Calculate required noise power based on target SNR
        snr_linear = 10 ** (snr_db / 10.0)
        desired_noise_power = signal_power / snr_linear
        
        # Adjust noise amplitude to meet the target power
        current_noise_power = torch.mean(pink_noise ** 2)
        if current_noise_power > 0:
            scale_factor = torch.sqrt(desired_noise_power / current_noise_power)
            scaled_noise = pink_noise * scale_factor
        else:
            scaled_noise = pink_noise * torch.sqrt(desired_noise_power)
        
        # Add noise to the signal
        scaled_noise = scaled_noise.unsqueeze(0).unsqueeze(0)
        noisy_audio = audio_tensor + scaled_noise
        
        # Convert back to numpy and normalize if clipping occurs
        noisy_audio_np = noisy_audio.squeeze().cpu().numpy()
        max_val = np.max(np.abs(noisy_audio_np))
        if max_val > 1.0:
            noisy_audio_np = noisy_audio_np / max_val * 0.95
            
        return noisy_audio_np
    
    @staticmethod
    def lowpass_filter_3k(
        audio_np: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Applies a low-pass filter with a 3kHz cutoff frequency.
        Simulates band-limited transmission scenarios.
        """
        audio_tensor = torch.from_numpy(audio_np).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        
        audio_tensor = audio_tensor.to(device)
        
        # Apply filter using julius library
        cutoff_freq = 3000
        filtered_audio = julius.lowpass_filter(audio_tensor, cutoff=cutoff_freq / sample_rate)
        
        filtered_audio_np = filtered_audio.squeeze().cpu().numpy()
        max_val = np.max(np.abs(filtered_audio_np))
        if max_val > 1.0:
            filtered_audio_np = filtered_audio_np / max_val * 0.95
            
        return filtered_audio_np
    
    @staticmethod
    def bandpass_filter_300_8k(
        audio_np: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Applies a band-pass filter (300Hz to 8kHz).
        Simulates standard telephony or restricted bandwidth channels.
        """
        audio_tensor = torch.from_numpy(audio_np).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        
        audio_tensor = audio_tensor.to(device)
        
        cutoff_freq_low = 300
        cutoff_freq_high = 8000
        filtered_audio = julius.bandpass_filter(
            audio_tensor,
            cutoff_low=cutoff_freq_low / sample_rate,
            cutoff_high=cutoff_freq_high / sample_rate
        )
        
        filtered_audio_np = filtered_audio.squeeze().cpu().numpy()
        max_val = np.max(np.abs(filtered_audio_np))
        if max_val > 1.0:
            filtered_audio_np = filtered_audio_np / max_val * 0.95
            
        return filtered_audio_np
    
    @staticmethod
    def stretch_2(
        audio_np: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Applies a 2x speed stretch. 
        Note: Stretching changes the audio length; this method resamples and pads/crops 
        to maintain the original buffer size.
        """
        audio_tensor = torch.from_numpy(audio_np).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        
        audio_tensor = audio_tensor.to(device)
        
        speed = 2.0
        new_sr = int(sample_rate * 1 / speed)
        
        # Resample to change speed
        stretched_audio = julius.resample_frac(audio_tensor, sample_rate, new_sr)
        
        # Ensure output length matches original input length
        target_length = audio_tensor.shape[-1]
        if stretched_audio.shape[-1] > target_length:
            stretched_audio = stretched_audio[..., :target_length]
        elif stretched_audio.shape[-1] < target_length:
            pad_length = target_length - stretched_audio.shape[-1]
            stretched_audio = torch.nn.functional.pad(stretched_audio, (0, pad_length))
        
        stretched_audio_np = stretched_audio.squeeze().cpu().numpy()
        max_val = np.max(np.abs(stretched_audio_np))
        if max_val > 1.0:
            stretched_audio_np = stretched_audio_np / max_val * 0.95
            
        return stretched_audio_np
    
    @staticmethod
    def cropping_front_back(
        audio_np: np.ndarray,
        crop_ratio: float = 0.1,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Crops a percentage of the audio from both the beginning and the end.
        Fills the cropped sections with zeros to keep the total length constant.
        """
        audio_length = len(audio_np)
        crop_samples = int(audio_length * crop_ratio)
        
        cropped_audio = np.zeros_like(audio_np)
        
        if crop_samples * 2 >= audio_length:
            # Fallback for extreme cropping: keep only a small center portion
            keep_samples = max(1, audio_length // 10)
            start = (audio_length - keep_samples) // 2
            cropped_audio[start:start+keep_samples] = audio_np[start:start+keep_samples]
        else:
            # Standard crop: keep the middle section
            cropped_audio[crop_samples:-crop_samples] = audio_np[crop_samples:-crop_samples]
        
        # Prevent clipping
        max_val = np.max(np.abs(cropped_audio))
        if max_val > 1.0:
            cropped_audio = cropped_audio / max_val * 0.95
            
        return cropped_audio
    
    @staticmethod
    def echo_default(
        audio_np: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Simulates an echo effect using an impulse response convolution.
        Default parameters: volume = 0.3, delay duration = 0.3s.
        """
        audio_tensor = torch.from_numpy(audio_np).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
        
        audio_tensor = audio_tensor.to(device)
        
        volume = 0.3
        duration = 0.3
        n_samples = int(sample_rate * duration)
        
        # Create a simple echo impulse response
        impulse_response = torch.zeros(n_samples).to(device)
        impulse_response[0] = 1.0  # Original signal
        if n_samples > 0:
            impulse_response[-1] = volume  # Delayed echo
        
        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)
        
        # Apply convolution for echo effect
        echoed_audio = julius.fft_conv1d(audio_tensor, impulse_response)
        
        # Match lengths
        if echoed_audio.shape[-1] > audio_tensor.shape[-1]:
            echoed_audio = echoed_audio[..., :audio_tensor.shape[-1]]
        elif echoed_audio.shape[-1] < audio_tensor.shape[-1]:
            pad_length = audio_tensor.shape[-1] - echoed_audio.shape[-1]
            echoed_audio = torch.nn.functional.pad(echoed_audio, (0, pad_length))
        
        echoed_audio_np = echoed_audio.squeeze().cpu().numpy()
        max_val = np.max(np.abs(echoed_audio_np))
        if max_val > 1.0:
            echoed_audio_np = echoed_audio_np / max_val * 0.95
            
        return echoed_audio_np
    
    @staticmethod
    def random_noise_attack(
        audio_np: np.ndarray,
        snr_db: float = 20.0,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Adds Gaussian White Noise (AWGN) for comparison against pink noise.
        """
        signal_power = np.mean(audio_np ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Generate AWGN
        noise = np.random.normal(0, np.sqrt(noise_power), audio_np.shape)
        
        noisy_audio = audio_np + noise
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
        
        return noisy_audio