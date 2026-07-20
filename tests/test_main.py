import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from src.main import run_pipeline, cli
from src.config import settings


@pytest.fixture(autouse=True)
def mock_output_dir(tmp_path, monkeypatch):
    """Overrides workspace directories to a temporary location during tests."""
    monkeypatch.setattr(settings, "RAW_AUDIO_DIR", tmp_path / "raw")
    monkeypatch.setattr(settings, "NORMALIZED_AUDIO_DIR", tmp_path / "normalized")
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "output")


@patch("src.main.validate_audio")
@patch("src.main.normalize_audio")
@patch("src.main.transcribe_audio")
@patch("src.main.diarize_audio")
@patch("src.main.merge_transcript_and_diarization")
@patch("src.main.run_benchmark")
def test_run_pipeline_success(
    mock_benchmark,
    mock_merge,
    mock_diarize,
    mock_transcribe,
    mock_normalize,
    mock_validate,
    tmp_path,
):
    """Verify that run_pipeline executes all stages sequentially and returns merged segments."""
    # Setup mock returns
    mock_validate.return_value = {"duration": 12.5}
    mock_normalize.return_value = tmp_path / "normalized" / "input_normalized.wav"
    mock_transcribe.return_value = {"segments": []}
    mock_diarize.return_value = []
    mock_merge.return_value = [{"speaker": "Speaker 1", "text": "Hello"}]

    input_file = tmp_path / "input.mp3"
    input_file.touch()

    result = run_pipeline(input_file)

    assert result == [{"speaker": "Speaker 1", "text": "Hello"}]
    mock_validate.assert_called_once_with(input_file)
    mock_normalize.assert_called_once()
    mock_transcribe.assert_called_once()
    mock_diarize.assert_called_once()
    mock_merge.assert_called_once()
    mock_benchmark.assert_called_once()


@patch("src.main.run_pipeline")
def test_cli_execution_success(mock_run_pipeline):
    """Verify that running the cli directly parses CLI args and executes run_pipeline."""
    mock_run_pipeline.return_value = [
        {"speaker": "Speaker 1", "start": 0.0, "text": "Hello"}
    ]

    # Mock system argv
    test_args = ["main.py", "--audio", "test_file.mp3"]
    with patch.object(sys, "argv", test_args):
        cli()

    mock_run_pipeline.assert_called_once_with(Path("test_file.mp3"))


@patch("src.main.run_pipeline")
def test_cli_execution_failure(mock_run_pipeline):
    """Verify CLI exit on pipeline errors."""
    mock_run_pipeline.side_effect = Exception("FFmpeg failed")

    test_args = ["main.py", "--audio", "test_file.mp3"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            cli()

        assert exc_info.value.code == 1
