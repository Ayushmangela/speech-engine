import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from src.normalize import (
    validate_audio,
    normalize_audio,
    AudioValidationError,
    AudioNormalizationError,
)
from src.config import settings


@pytest.fixture(autouse=True)
def mock_output_dir(tmp_path, monkeypatch):
    """Overrides outputs directory setting to a temp directory during tests."""
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(settings, "NORMALIZED_AUDIO_DIR", tmp_path / "normalized")


def test_validate_audio_non_existent():
    """Verify that a validation error is raised if the file does not exist."""
    with pytest.raises(AudioValidationError, match="Audio file does not exist"):
        validate_audio(Path("non_existent_file.mp3"))


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
def test_validate_audio_not_a_file(mock_is_file, mock_exists):
    """Verify that a validation error is raised if the path is not a file."""
    mock_exists.return_value = True
    mock_is_file.return_value = False
    with pytest.raises(AudioValidationError, match="Path is not a file"):
        validate_audio(Path("directory_path"))


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
def test_validate_audio_unsupported_format(mock_is_file, mock_exists):
    """Verify that validation fails for unsupported extensions."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        validate_audio(Path("test.txt"))


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
def test_validate_audio_too_large(mock_stat, mock_is_file, mock_exists):
    """Verify validation fails if the file size exceeds configured limits."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value.st_size = settings.MAX_FILE_SIZE_BYTES + 100
    with pytest.raises(AudioValidationError, match="exceeds maximum allowed size"):
        validate_audio(Path("test.mp3"))


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
@patch("subprocess.run")
def test_validate_audio_ffprobe_corrupted(
    mock_run, mock_stat, mock_is_file, mock_exists
):
    """Verify validation fails if ffprobe exits with an error (indicating corruption)."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value.st_size = 1000

    # Mock subprocess.run raising CalledProcessError
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="ffprobe", stderr="Corrupted file content"
    )

    with pytest.raises(AudioValidationError, match="File is corrupted or unreadable"):
        validate_audio(Path("corrupted.mp3"))


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
@patch("subprocess.run")
def test_validate_audio_ffprobe_not_installed(
    mock_run, mock_stat, mock_is_file, mock_exists
):
    """Verify validation fails if ffprobe is not installed on the system."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value.st_size = 1000

    mock_run.side_effect = FileNotFoundError()

    with pytest.raises(AudioValidationError, match="ffprobe is not installed"):
        validate_audio(Path("test.mp3"))


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
@patch("subprocess.run")
def test_validate_audio_invalid_duration(
    mock_run, mock_stat, mock_is_file, mock_exists
):
    """Verify validation fails if audio duration is 0 or negative."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value.st_size = 1000

    # Mock ffprobe output returning duration=0
    probe_output = {
        "format": {"duration": "0"},
        "streams": [{"codec_name": "mp3", "sample_rate": "44100", "channels": 2}],
    }
    mock_response = MagicMock()
    mock_response.stdout = json.dumps(probe_output)
    mock_run.return_value = mock_response

    with pytest.raises(AudioValidationError, match="Invalid audio duration"):
        validate_audio(Path("silent.mp3"))


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
@patch("subprocess.run")
def test_validate_audio_too_long(mock_run, mock_stat, mock_is_file, mock_exists):
    """Verify validation fails if duration exceeds the limit in configuration."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value.st_size = 1000

    probe_output = {
        "format": {"duration": str(settings.MAX_AUDIO_DURATION_SECONDS + 10)},
        "streams": [{"codec_name": "mp3", "sample_rate": "44100", "channels": 2}],
    }
    mock_response = MagicMock()
    mock_response.stdout = json.dumps(probe_output)
    mock_run.return_value = mock_response

    with pytest.raises(AudioValidationError, match="exceeds maximum allowed duration"):
        validate_audio(Path("long.mp3"))


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
@patch("subprocess.run")
def test_validate_audio_success(
    mock_run, mock_stat, mock_is_file, mock_exists, tmp_path
):
    """Verify metadata generation and successful validation of a valid file."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value.st_size = 50000

    probe_output = {
        "format": {"duration": "120.5"},
        "streams": [
            {
                "codec_name": "mp3",
                "sample_rate": "44100",
                "channels": 2,
                "bit_rate": "192000",
            }
        ],
    }
    mock_response = MagicMock()
    mock_response.stdout = json.dumps(probe_output)
    mock_run.return_value = mock_response

    metadata = validate_audio(Path("valid.mp3"))

    assert metadata["filename"] == "valid.mp3"
    assert metadata["duration"] == 120.5
    assert metadata["channels"] == 2
    assert metadata["sample_rate"] == 44100
    assert metadata["bitrate"] == 192000
    assert metadata["codec"] == "mp3"

    # Check that metadata file was written
    metadata_file = settings.OUTPUT_DIR / "audio_metadata.json"
    assert metadata_file.exists()
    with open(metadata_file, "r") as f:
        saved_data = json.load(f)
    assert saved_data["filename"] == "valid.mp3"


@patch("src.normalize.validate_audio")
@patch("subprocess.run")
def test_normalize_audio_success(mock_run, mock_validate, tmp_path):
    """Verify that normalize_audio invokes FFmpeg with expected params."""
    mock_validate.return_value = {}

    input_file = Path("input.mp3")
    output_dir = tmp_path / "normalized"

    normalized_file = normalize_audio(input_file, output_dir=output_dir)

    assert normalized_file.name == "input_normalized.wav"
    assert normalized_file.parent == output_dir

    # Assert ffmpeg was called
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "ffmpeg"
    assert "-ar" in cmd
    assert "16000" in cmd
    assert "-ac" in cmd
    assert "1" in cmd
    assert "-c:a" in cmd
    assert "pcm_s16le" in cmd


@patch("src.normalize.validate_audio")
@patch("subprocess.run")
def test_normalize_audio_ffmpeg_error(mock_run, mock_validate):
    """Verify error wrapping if FFmpeg execution fails."""
    mock_validate.return_value = {}
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd="ffmpeg", stderr="Conversion error"
    )

    with pytest.raises(AudioNormalizationError, match="FFmpeg normalization failed"):
        normalize_audio(Path("input.mp3"))


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
@patch("subprocess.run")
def test_validate_audio_no_format(mock_run, mock_stat, mock_is_file, mock_exists):
    """Verify validation fails if format info is missing in ffprobe output."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value.st_size = 1000

    probe_output = {
        # Format key is missing
        "streams": [{"codec_name": "mp3", "sample_rate": "44100", "channels": 2}]
    }
    mock_response = MagicMock()
    mock_response.stdout = json.dumps(probe_output)
    mock_run.return_value = mock_response

    with pytest.raises(
        AudioValidationError, match="No audio streams or format info found"
    ):
        validate_audio(Path("no_format.mp3"))


@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
@patch("subprocess.run")
def test_validate_audio_value_error_duration(
    mock_run, mock_stat, mock_is_file, mock_exists
):
    """Verify validation fails if duration string is not parseable to float."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value.st_size = 1000

    probe_output = {
        "format": {"duration": "invalid_duration_string"},
        "streams": [{"codec_name": "mp3", "sample_rate": "44100", "channels": 2}],
    }
    mock_response = MagicMock()
    mock_response.stdout = json.dumps(probe_output)
    mock_run.return_value = mock_response

    with pytest.raises(AudioValidationError, match="Invalid audio duration"):
        validate_audio(Path("invalid.mp3"))


@patch("src.normalize.validate_audio")
@patch("subprocess.run")
def test_normalize_audio_ffmpeg_not_found(mock_run, mock_validate):
    """Verify error wrapping if ffmpeg command is not found."""
    mock_validate.return_value = {}
    mock_run.side_effect = FileNotFoundError()

    with pytest.raises(
        AudioNormalizationError, match="ffmpeg is not installed or not in PATH"
    ):
        normalize_audio(Path("input.mp3"))
