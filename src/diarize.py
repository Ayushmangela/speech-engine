import json
from pathlib import Path
from typing import Dict, Any, List
from src.config import settings
from src.logger import logger
from src.registry import ModelRegistry


def diarize_audio(file_path: Path) -> List[Dict[str, Any]]:
    """
    Runs speaker diarization on the audio file using pyannote.audio.
    Saves raw diarization turns to output/raw_diarization.json.
    Renames speakers sequentially (e.g. Speaker 1, Speaker 2) based on chronological order of appearance.

    Args:
        file_path: Path to the normalized WAV audio file.

    Returns:
        List[Dict[str, Any]]: List of speaker segments with start, end, and mapped speaker.
    """
    logger.info(f"Starting speaker diarization for {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Normalized audio file not found: {file_path}")

    # Lazy-load diarization pipeline from registry
    pipeline = ModelRegistry.get_diarization_pipeline()

    logger.info("Invoking pyannote speaker diarization...")
    try:
        diarization_result = pipeline(str(file_path))
    except Exception as e:
        logger.exception("Diarization execution failed")
        raise RuntimeError(f"Diarization pipeline failed: {e}") from e

    raw_turns = []
    speaker_map: Dict[str, str] = {}
    next_speaker_num = 1

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        # Map SPEAKER_00, SPEAKER_01 -> Speaker 1, Speaker 2 sequentially
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {next_speaker_num}"
            next_speaker_num += 1

        mapped_speaker = speaker_map[speaker]

        raw_turns.append(
            {
                "start": round(float(turn.start), 2),
                "end": round(float(turn.end), 2),
                "speaker": mapped_speaker,
                "raw_speaker": speaker,
            }
        )

    # Save output/raw_diarization.json
    try:
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        raw_diarization_file = settings.OUTPUT_DIR / "raw_diarization.json"
        with open(raw_diarization_file, "w", encoding="utf-8") as f:
            json.dump(raw_turns, f, indent=2)
        logger.info(f"Raw diarization saved to {raw_diarization_file}")
    except Exception as e:
        logger.exception("Failed to save raw diarization JSON")
        raise RuntimeError(f"Could not save raw_diarization.json: {e}") from e

    return raw_turns
