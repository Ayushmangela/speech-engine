import json
from typing import Dict, Any, List, Optional
from src.config import settings
from src.logger import logger
from src.transcribe import format_timestamp


def get_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """
    Calculates overlap duration between two time intervals.

    Args:
        start1: Start time of first interval.
        end1: End time of first interval.
        start2: Start time of second interval.
        end2: End time of second interval.

    Returns:
        float: Overlap duration in seconds.
    """
    return max(0.0, min(end1, end2) - max(start1, start2))


def get_distance(start1: float, end1: float, start2: float, end2: float) -> float:
    """
    Calculates the temporal distance between two intervals.
    Returns 0.0 if they overlap.

    Args:
        start1: Start time of first interval.
        end1: End time of first interval.
        start2: Start time of second interval.
        end2: End time of second interval.

    Returns:
        float: Distance in seconds.
    """
    if end1 <= start2:
        return start2 - end1
    if end2 <= start1:
        return start1 - end2
    return 0.0


def assign_speaker(
    start: float, end: float, speaker_turns: List[Dict[str, Any]]
) -> str:
    """
    Assigns a speaker to a time interval [start, end].
    First checks for maximum overlap. If no overlap, falls back to the closest speaker.
    If no speaker turns are available, returns "Unknown".

    Args:
        start: Interval start.
        end: Interval end.
        speaker_turns: List of diarized speaker turns.

    Returns:
        str: Mapped speaker name.
    """
    if not speaker_turns:
        return "Unknown"

    best_speaker = None
    max_overlap = 0.0

    # Try to find the speaker with maximum overlap
    for turn in speaker_turns:
        overlap = get_overlap(start, end, turn["start"], turn["end"])
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = turn["speaker"]

    if max_overlap > 0.0:
        return best_speaker

    # If no overlapping speaker was found, find the closest one
    min_distance = float("inf")
    closest_speaker = "Unknown"

    for turn in speaker_turns:
        dist = get_distance(start, end, turn["start"], turn["end"])
        if dist < min_distance:
            min_distance = dist
            closest_speaker = turn["speaker"]

    return closest_speaker


def merge_transcript_and_diarization(
    whisper_result: Dict[str, Any], diarization_result: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Merges transcription segments and word-level timestamps with speaker diarization turns.
    Saves merged segments to output/merged.json and text transcript to output/transcript.txt.

    Args:
        whisper_result: Dictionary parsed from raw_whisper.json.
        diarization_result: List of dictionaries parsed from raw_diarization.json.

    Returns:
        List[Dict[str, Any]]: List of merged speaker segments.
    """
    logger.info("Starting merge of transcription and diarization results")

    raw_segments = whisper_result.get("segments", [])
    merged_items = []

    # Map words or segments to speakers
    for seg in raw_segments:
        words = seg.get("words", [])
        if words:
            # Word-level matching
            for word_info in words:
                start = word_info["start"]
                end = word_info["end"]
                word_text = word_info["word"].strip()
                if not word_text:
                    continue
                prob = word_info.get("probability", 1.0)

                speaker = assign_speaker(start, end, diarization_result)
                merged_items.append(
                    {
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                        "text": word_text,
                        "confidence": prob,
                    }
                )
        else:
            # Segment-level fallback
            start = seg["start"]
            end = seg["end"]
            text = seg["text"].strip()
            conf = seg.get("confidence", 1.0)

            speaker = assign_speaker(start, end, diarization_result)
            merged_items.append(
                {
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": text,
                    "confidence": conf,
                }
            )

    if not merged_items:
        logger.info("No transcription items to merge.")
        return []

    # Group consecutive items with the same speaker
    grouped_segments = []
    current_group: Optional[Dict[str, Any]] = None

    for item in merged_items:
        if current_group is None:
            current_group = {
                "speaker": item["speaker"],
                "start": item["start"],
                "end": item["end"],
                "items": [item],
            }
        elif item["speaker"] == current_group["speaker"]:
            # Extend current speaker group
            current_group["end"] = item["end"]
            current_group["items"].append(item)
        else:
            # Speaker changed, finalize current group and start new one
            grouped_segments.append(current_group)
            current_group = {
                "speaker": item["speaker"],
                "start": item["start"],
                "end": item["end"],
                "items": [item],
            }

    if current_group:
        grouped_segments.append(current_group)

    # Format grouped segments, calculate average confidence, and apply low confidence marker
    final_segments = []
    for group in grouped_segments:
        texts = []
        confidences = []
        for item in group["items"]:
            texts.append(item["text"])
            confidences.append(item["confidence"])

        # Whisper word-level output might contain spaces or punctuation, so join and clean spaces
        joined_text = " ".join(texts).strip()
        avg_conf = sum(confidences) / len(confidences) if confidences else 1.0

        # Check low confidence threshold
        if (
            avg_conf < settings.WHISPER_CONFIDENCE_THRESHOLD
            and not joined_text.startswith("[Low Confidence]")
        ):
            joined_text = f"[Low Confidence] {joined_text}"

        final_segments.append(
            {
                "speaker": group["speaker"],
                "start": round(group["start"], 2),
                "end": round(group["end"], 2),
                "text": joined_text,
                "confidence": round(avg_conf, 2),
            }
        )

    # Save output/merged.json
    try:
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        merged_file = settings.OUTPUT_DIR / "merged.json"
        with open(merged_file, "w", encoding="utf-8") as f:
            json.dump(final_segments, f, indent=2)
        logger.info(f"Merged output saved to {merged_file}")
    except Exception as e:
        logger.exception("Failed to save merged.json")
        raise RuntimeError(f"Could not save merged.json: {e}") from e

    # Save output/transcript.txt formatted with speakers and timestamps
    try:
        transcript_lines = []
        for seg in final_segments:
            timestamp_str = format_timestamp(seg["start"])
            transcript_lines.append(
                f"{timestamp_str}\n\n{seg['speaker']}:\n{seg['text']}\n"
            )

        transcript_file = settings.OUTPUT_DIR / "transcript.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))
        logger.info(f"Merged transcript text saved to {transcript_file}")
    except Exception as e:
        logger.exception("Failed to save transcript.txt")
        raise RuntimeError(f"Could not save transcript.txt: {e}") from e

    return final_segments
