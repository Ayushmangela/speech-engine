import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from src.diarize import diarize_audio
from src.config import settings


class MockSegment:
    """Mock representing pyannote Segment with start and end times."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class MockAnnotation:
    """Mock representing pyannote Annotation result."""

    def itertracks(self, yield_label: bool = True):
        # Speaker 01 speaks first -> should map to Speaker 1
        # Speaker 00 speaks second -> should map to Speaker 2
        yield MockSegment(0.5, 3.2), None, "SPEAKER_01"
        yield MockSegment(3.2, 5.0), None, "SPEAKER_00"
        yield MockSegment(5.1, 7.5), None, "SPEAKER_01"
        yield MockSegment(
            7.5, 10.0
        ), None, "SPEAKER_02"  # Speaker 02 speaks third -> Speaker 3


@pytest.fixture(autouse=True)
def mock_output_dir(tmp_path, monkeypatch):
    """Overrides outputs directory setting to a temp directory during tests."""
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "output")


@patch("src.diarize.ModelRegistry.get_diarization_pipeline")
def test_diarize_audio_not_found(mock_get_pipeline):
    """Verify FileNotFoundError is raised if audio path does not exist."""
    with pytest.raises(FileNotFoundError):
        diarize_audio(Path("non_existent.wav"))


@patch("src.diarize.ModelRegistry.get_diarization_pipeline")
@patch("pathlib.Path.exists")
def test_diarize_audio_success(mock_exists, mock_get_pipeline):
    """Verify diarization pipeline maps speakers sequentially and saves output file."""
    mock_exists.return_value = True

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = MockAnnotation()
    mock_get_pipeline.return_value = mock_pipeline

    input_file = Path("test_normalized.wav")
    result = diarize_audio(input_file)

    # Verify return list size
    assert len(result) == 4

    # Chronological mapping assertions:
    # SPEAKER_01 -> Speaker 1
    assert result[0]["speaker"] == "Speaker 1"
    assert result[0]["raw_speaker"] == "SPEAKER_01"
    assert result[0]["start"] == 0.5
    assert result[0]["end"] == 3.2

    # SPEAKER_00 -> Speaker 2
    assert result[1]["speaker"] == "Speaker 2"
    assert result[1]["raw_speaker"] == "SPEAKER_00"

    # SPEAKER_01 -> Speaker 1 (previously mapped)
    assert result[2]["speaker"] == "Speaker 1"

    # SPEAKER_02 -> Speaker 3
    assert result[3]["speaker"] == "Speaker 3"
    assert result[3]["raw_speaker"] == "SPEAKER_02"

    # Verify JSON output was written
    raw_diarization_file = settings.OUTPUT_DIR / "raw_diarization.json"
    assert raw_diarization_file.exists()

    with open(raw_diarization_file, "r") as f:
        saved_data = json.load(f)

    assert len(saved_data) == 4
    assert saved_data[0]["speaker"] == "Speaker 1"
    assert saved_data[1]["speaker"] == "Speaker 2"
