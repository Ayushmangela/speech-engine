import pytest
from src.merge import (
    get_overlap,
    get_distance,
    assign_speaker,
    merge_transcript_and_diarization,
)
from src.config import settings


@pytest.fixture(autouse=True)
def mock_output_dir(tmp_path, monkeypatch):
    """Overrides outputs directory setting to a temp directory during tests."""
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "output")


def test_get_overlap():
    """Verify that overlap durations are calculated correctly."""
    assert get_overlap(0.0, 5.0, 2.0, 4.0) == 2.0
    assert get_overlap(0.0, 5.0, 4.0, 6.0) == 1.0
    assert get_overlap(0.0, 2.0, 3.0, 5.0) == 0.0
    assert get_overlap(10.0, 15.0, 8.0, 12.0) == 2.0


def test_get_distance():
    """Verify that distances between non-overlapping intervals are calculated correctly."""
    assert get_distance(0.0, 2.0, 3.0, 5.0) == 1.0
    assert get_distance(3.0, 5.0, 0.0, 2.0) == 1.0
    assert get_distance(0.0, 5.0, 2.0, 4.0) == 0.0  # Overlapping has 0.0 distance


def test_assign_speaker_empty():
    """Verify that assign_speaker returns 'Unknown' if no speaker turns exist."""
    assert assign_speaker(0.0, 5.0, []) == "Unknown"


def test_assign_speaker_overlap():
    """Verify that assign_speaker chooses the speaker with the maximum overlap."""
    turns = [
        {"speaker": "Speaker 1", "start": 0.0, "end": 2.0},
        {"speaker": "Speaker 2", "start": 2.0, "end": 6.0},
        {"speaker": "Speaker 3", "start": 6.0, "end": 10.0},
    ]
    # Interval 1.0 to 4.0 has:
    # 1.0s overlap with Speaker 1 (1.0 to 2.0)
    # 2.0s overlap with Speaker 2 (2.0 to 4.0)
    # Speaker 2 has maximum overlap
    assert assign_speaker(1.0, 4.0, turns) == "Speaker 2"


def test_assign_speaker_closest_fallback():
    """Verify that assign_speaker falls back to the closest speaker if no overlap exists."""
    turns = [
        {"speaker": "Speaker 1", "start": 1.0, "end": 2.0},
        {"speaker": "Speaker 2", "start": 5.0, "end": 6.0},
    ]
    # Interval 3.0 to 4.0 has no overlap with Speaker 1 (dist = 1.0) or Speaker 2 (dist = 1.0).
    # Interval 3.0 to 3.5 is closer to Speaker 1 (dist = 1.0) than Speaker 2 (dist = 1.5).
    assert assign_speaker(3.0, 3.5, turns) == "Speaker 1"
    # Interval 4.5 to 4.8 is closer to Speaker 2 (dist = 0.2) than Speaker 1 (dist = 2.5).
    assert assign_speaker(4.5, 4.8, turns) == "Speaker 2"


def test_merge_transcript_and_diarization_word_level():
    """Verify merging using word-level timestamps."""
    whisper_data = {
        "segments": [
            {
                "start": 0.0,
                "end": 4.5,
                "text": "Hello world this is a test.",
                "confidence": 0.95,
                "words": [
                    {"start": 0.1, "end": 0.5, "word": "Hello", "probability": 0.98},
                    {"start": 0.6, "end": 1.2, "word": "world", "probability": 0.95},
                    {"start": 1.5, "end": 2.2, "word": "this", "probability": 0.92},
                    {"start": 2.3, "end": 3.0, "word": "is", "probability": 0.94},
                    {"start": 3.1, "end": 3.8, "word": "a", "probability": 0.96},
                    {"start": 3.9, "end": 4.4, "word": "test.", "probability": 0.97},
                ],
            }
        ]
    }

    diarization_data = [
        {"speaker": "Speaker 1", "start": 0.0, "end": 1.3},
        {"speaker": "Speaker 2", "start": 1.4, "end": 4.5},
    ]

    result = merge_transcript_and_diarization(whisper_data, diarization_data)

    # Expected output:
    # Speaker 1 speaks "Hello world"
    # Speaker 2 speaks "this is a test."
    assert len(result) == 2

    assert result[0]["speaker"] == "Speaker 1"
    assert result[0]["text"] == "Hello world"
    assert result[0]["start"] == 0.1
    assert result[0]["end"] == 1.2

    assert result[1]["speaker"] == "Speaker 2"
    assert result[1]["text"] == "this is a test."
    assert result[1]["start"] == 1.5
    assert result[1]["end"] == 4.4

    # Check outputs generated
    merged_file = settings.OUTPUT_DIR / "merged.json"
    transcript_file = settings.OUTPUT_DIR / "transcript.txt"
    assert merged_file.exists()
    assert transcript_file.exists()

    with open(transcript_file, "r") as f:
        content = f.read()
    assert "Speaker 1:\nHello world" in content
    assert "Speaker 2:\nthis is a test." in content


def test_merge_transcript_and_diarization_segment_fallback():
    """Verify fallback to segment-level merging if word-level is missing."""
    whisper_data = {
        "segments": [
            {
                "start": 0.5,
                "end": 2.0,
                "text": "Hello world.",
                "confidence": 0.98,
                "words": None,
            },
            {
                "start": 2.5,
                "end": 4.5,
                "text": "How are you?",
                "confidence": 0.92,
                "words": None,
            },
        ]
    }

    diarization_data = [
        {"speaker": "Speaker 1", "start": 0.0, "end": 2.2},
        {"speaker": "Speaker 2", "start": 2.3, "end": 5.0},
    ]

    result = merge_transcript_and_diarization(whisper_data, diarization_data)

    assert len(result) == 2
    assert result[0]["speaker"] == "Speaker 1"
    assert result[0]["text"] == "Hello world."
    assert result[1]["speaker"] == "Speaker 2"
    assert result[1]["text"] == "How are you?"


def test_merge_transcript_and_diarization_consecutive_merge():
    """Verify that consecutive segments by the same speaker are merged."""
    whisper_data = {
        "segments": [
            {
                "start": 0.5,
                "end": 2.0,
                "text": "Hello world.",
                "confidence": 0.98,
                "words": None,
            },
            {
                "start": 2.1,
                "end": 4.0,
                "text": "How are you?",
                "confidence": 0.95,
                "words": None,
            },
        ]
    }

    # Both segments assigned to Speaker 1
    diarization_data = [{"speaker": "Speaker 1", "start": 0.0, "end": 5.0}]

    result = merge_transcript_and_diarization(whisper_data, diarization_data)

    assert len(result) == 1
    assert result[0]["speaker"] == "Speaker 1"
    assert result[0]["text"] == "Hello world. How are you?"
    assert result[0]["start"] == 0.5
    assert result[0]["end"] == 4.0


def test_merge_transcript_and_diarization_low_confidence():
    """Verify prefixing of low confidence merged segments."""
    whisper_data = {
        "segments": [
            {
                "start": 0.5,
                "end": 2.0,
                "text": "Garbled speech.",
                # confidence = 0.3 (below 0.6)
                "confidence": 0.3,
                "words": None,
            }
        ]
    }

    diarization_data = [{"speaker": "Speaker 1", "start": 0.0, "end": 3.0}]

    result = merge_transcript_and_diarization(whisper_data, diarization_data)

    assert len(result) == 1
    assert result[0]["text"].startswith("[Low Confidence]")


def test_merge_transcript_empty():
    """Verify that merging an empty whisper result returns an empty list."""
    result = merge_transcript_and_diarization({"segments": []}, [])
    assert result == []
