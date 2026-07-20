import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from src.config import settings
from src.logger import logger


class AudioValidationError(Exception):
    """Exception raised when audio validation fails."""

    pass


class AudioNormalizationError(Exception):
    """Exception raised when audio normalization fails."""

    pass


def validate_audio(file_path: Path) -> Dict[str, Any]:
    """
    Validates the audio file for existence, size, readability, duration, and corruption.
    Generates and returns metadata, saving it to output/audio_metadata.json.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dict[str, Any]: Extracted metadata.

    Raises:
        AudioValidationError: If validation fails.
    """
    logger.info(f"Starting validation for: {file_path}")

    if not file_path.exists():
        raise AudioValidationError(f"Audio file does not exist: {file_path}")

    if not file_path.is_file():
        raise AudioValidationError(f"Path is not a file: {file_path}")

    # Check extension
    supported_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
    ext = file_path.suffix.lower()
    if ext not in supported_extensions:
        raise AudioValidationError(
            f"Unsupported audio format: '{ext}'. Supported formats: {sorted(list(supported_extensions))}"
        )

    # Check size
    file_size = file_path.stat().st_size
    if file_size > settings.MAX_FILE_SIZE_BYTES:
        max_mb = settings.MAX_FILE_SIZE_BYTES / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        raise AudioValidationError(
            f"File size ({actual_mb:.2f} MB) exceeds maximum allowed size ({max_mb:.2f} MB)"
        )

    # Run ffprobe to get metadata and check readability/corruption
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bit_rate:format=duration,size",
        "-of",
        "json",
        str(file_path),
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        logger.exception(f"ffprobe check failed for {file_path}")
        raise AudioValidationError(
            f"File is corrupted or unreadable by FFmpeg. Stderr: {e.stderr}"
        ) from e
    except FileNotFoundError as e:
        logger.exception("ffprobe command not found on system path")
        raise AudioValidationError("ffprobe is not installed or not in PATH.") from e

    try:
        probe_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse ffprobe JSON output")
        raise AudioValidationError("Invalid metadata returned by ffprobe") from e

    if not probe_data or "format" not in probe_data:
        raise AudioValidationError(
            "No audio streams or format info found in file. It might be corrupted or silent."
        )

    fmt = probe_data.get("format", {})
    streams = probe_data.get("streams", [])

    if not streams:
        raise AudioValidationError("No audio streams detected in the file.")

    stream = streams[0]

    # Extract metadata fields
    try:
        duration = float(fmt.get("duration", 0))
    except (ValueError, TypeError):
        duration = 0.0

    if duration <= 0:
        raise AudioValidationError(
            f"Invalid audio duration: {duration} seconds. File might be empty or corrupted."
        )

    if duration > settings.MAX_AUDIO_DURATION_SECONDS:
        max_sec = settings.MAX_AUDIO_DURATION_SECONDS
        raise AudioValidationError(
            f"Audio duration ({duration:.2f}s) exceeds maximum allowed duration ({max_sec}s)"
        )

    try:
        sample_rate = int(stream.get("sample_rate", 0))
    except (ValueError, TypeError):
        sample_rate = 0

    try:
        channels = int(stream.get("channels", 0))
    except (ValueError, TypeError):
        channels = 0

    try:
        bitrate = int(stream.get("bit_rate", 0)) if stream.get("bit_rate") else None
    except (ValueError, TypeError):
        bitrate = None

    codec = stream.get("codec_name", "unknown")

    metadata = {
        "filename": file_path.name,
        "extension": ext[1:] if ext.startswith(".") else ext,
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": channels,
        "bitrate": bitrate,
        "file_size": file_size,
        "codec": codec,
    }

    # Save metadata to output/audio_metadata.json
    try:
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metadata_file = settings.OUTPUT_DIR / "audio_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_file}")
    except Exception as e:
        logger.exception("Failed to save metadata to disk")
        # Re-raise to ensure error handling standards are met
        raise AudioValidationError(f"Could not save metadata JSON: {e}") from e

    return metadata


def normalize_audio(input_path: Path, output_dir: Optional[Path] = None) -> Path:
    """
    Normalizes the input audio file to mono, 16kHz PCM WAV format.
    Saves the normalized file into output_dir (defaults to settings.NORMALIZED_AUDIO_DIR).
    Returns the path to the normalized file.

    Args:
        input_path: Path to the raw audio file.
        output_dir: Optional target directory. Defaults to settings.NORMALIZED_AUDIO_DIR.

    Returns:
        Path: Path to the normalized WAV file.

    Raises:
        AudioNormalizationError: If FFmpeg fails.
    """
    logger.info(f"Starting audio normalization for {input_path}")

    # First, validate the input file
    validate_audio(input_path)

    if output_dir is None:
        output_dir = settings.NORMALIZED_AUDIO_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename
    output_path = output_dir / f"{input_path.stem}_normalized.wav"

    # Run FFmpeg command to normalize
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        logger.info(f"Audio normalized successfully. Saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.exception(f"FFmpeg normalization failed for {input_path}")
        raise AudioNormalizationError(
            f"FFmpeg normalization failed. Stderr: {e.stderr}"
        ) from e
    except FileNotFoundError as e:
        logger.exception("ffmpeg command not found on system path")
        raise AudioNormalizationError("ffmpeg is not installed or not in PATH.") from e

    return output_path
