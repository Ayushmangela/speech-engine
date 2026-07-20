import json
import math
from pathlib import Path
from typing import Dict, Any
from src.config import settings
from src.logger import logger
from src.registry import ModelRegistry


def format_timestamp(seconds: float) -> str:
    """Formats duration in seconds to [HH:MM:SS] format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def transcribe_audio(file_path: Path) -> Dict[str, Any]:
    """
    Transcribes the normalized audio file using faster-whisper.
    Applies confidence thresholding and outputs raw_whisper.json and initial transcript.txt.

    Args:
        file_path: Path to the normalized WAV audio file.

    Returns:
        Dict[str, Any]: Transcription result dictionary.
    """
    logger.info(f"Starting audio transcription for {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Normalized audio file not found: {file_path}")

    # Get the Whisper model from registry
    model = ModelRegistry.get_whisper_model()

    # Run transcription
    logger.info("Invoking Whisper transcription...")
    try:
        segments, info = model.transcribe(
            str(file_path), word_timestamps=True, beam_size=5
        )
    except Exception as e:
        logger.exception("Whisper transcription execution failed")
        raise RuntimeError(f"Whisper transcription failed: {e}") from e

    # Materialize segments from generator
    processed_segments = []
    text_lines = []

    for segment in segments:
        # Convert log probability to a standard [0, 1] probability
        # avg_logprob is the average log probability of the tokens in the segment
        try:
            confidence = math.exp(segment.avg_logprob)
        except OverflowError:
            confidence = 0.0

        # Bound confidence to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))

        text = segment.text.strip()

        # Check confidence threshold
        is_low_confidence = confidence < settings.WHISPER_CONFIDENCE_THRESHOLD
        if is_low_confidence:
            text = f"[Low Confidence] {text}"
            logger.warning(
                f"Segment [{segment.start:.2f}s - {segment.end:.2f}s] below confidence threshold: "
                f"{confidence:.2f} < {settings.WHISPER_CONFIDENCE_THRESHOLD:.2f}"
            )

        # Extract word timestamps
        words_list = []
        if segment.words:
            for word in segment.words:
                words_list.append(
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                )

        processed_segments.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": text,
                "confidence": confidence,
                "words": words_list,
            }
        )

        # Formatting for the initial text transcript
        timestamp_str = format_timestamp(segment.start)
        text_lines.append(f"{timestamp_str}\n{text}\n")

    result = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": processed_segments,
    }

    # Save raw_whisper.json
    try:
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        raw_whisper_file = settings.OUTPUT_DIR / "raw_whisper.json"
        with open(raw_whisper_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Raw whisper transcript saved to {raw_whisper_file}")
    except Exception as e:
        logger.exception("Failed to save raw whisper JSON output")
        raise RuntimeError(f"Could not save raw_whisper.json: {e}") from e

    # Save initial transcript.txt (text only, without speakers for Milestone 1)
    try:
        transcript_txt_file = settings.OUTPUT_DIR / "transcript.txt"
        with open(transcript_txt_file, "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines))
        logger.info(f"Initial transcript text saved to {transcript_txt_file}")
    except Exception as e:
        logger.exception("Failed to save initial transcript TXT output")
        raise RuntimeError(f"Could not save transcript.txt: {e}") from e

    return result
