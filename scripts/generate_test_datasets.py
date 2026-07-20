import math
import struct
import wave
from pathlib import Path

def generate_sine_wave(
    file_path: Path,
    duration: float = 2.0,
    frequency: float = 440.0,
    sample_rate: int = 16000
) -> None:
    """
    Generates a valid mono 16-bit PCM WAV file containing a sine wave.
    
    Args:
        file_path: Output file path.
        duration: Audio duration in seconds.
        frequency: Audio wave frequency in Hz.
        sample_rate: Audio sampling rate in Hz.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration * sample_rate)
    
    with wave.open(str(file_path), "wb") as wav_file:
        # Mono (1 channel), 2 bytes per sample (16-bit), sample_rate, num_samples
        wav_file.setparams((1, 2, sample_rate, num_samples, "NONE", "not compressed"))
        
        for i in range(num_samples):
            val = math.sin(2.0 * math.pi * frequency * (i / sample_rate))
            sample = int(val * 32767)
            wav_file.writeframes(struct.pack("<h", sample))

def main() -> None:
    """Orchestrates test datasets creation."""
    base_dir = Path("datasets")
    
    generate_sine_wave(base_dir / "single_speaker" / "sample.wav", duration=2.0, frequency=440.0)
    generate_sine_wave(base_dir / "two_speaker" / "sample.wav", duration=5.0, frequency=600.0)
    generate_sine_wave(base_dir / "meeting" / "sample.wav", duration=10.0, frequency=800.0)
    generate_sine_wave(base_dir / "noise" / "sample.wav", duration=3.0, frequency=200.0)
    
    print("Test datasets generated successfully:")
    for path in base_dir.rglob("*.wav"):
        print(f" - {path}")

if __name__ == "__main__":
    main()
