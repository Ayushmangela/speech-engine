from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from src.transcribe import transcribe_audio, ModelRegistry, format_timestamp
from src.config import settings


@pytest.fixture(autouse=True)
def mock_output_dir(tmp_path, monkeypatch):
    """Overrides outputs directory setting to a temp directory during tests."""
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "output")


def test_format_timestamp():
    """Verify that formatting seconds into timestamps is correct."""
    assert format_timestamp(0) == "[00:00:00]"
    assert format_timestamp(5.2) == "[00:00:05]"
    assert format_timestamp(65.8) == "[00:01:05]"
    assert format_timestamp(3665) == "[01:01:05]"
    assert format_timestamp(86399) == "[23:59:59]"


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_device_detection_cuda(mock_mps, mock_cuda):
    """Verify that CUDA is selected if available."""
    mock_cuda.return_value = True
    mock_mps.return_value = False
    assert ModelRegistry.get_torch_device() == "cuda"


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_device_detection_mps(mock_mps, mock_cuda):
    """Verify that MPS is selected if CUDA is unavailable but MPS is available."""
    mock_cuda.return_value = False
    mock_mps.return_value = True
    assert ModelRegistry.get_torch_device() == "mps"


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_device_detection_cpu(mock_mps, mock_cuda):
    """Verify that CPU is selected if neither CUDA nor MPS is available."""
    mock_cuda.return_value = False
    mock_mps.return_value = False
    assert ModelRegistry.get_torch_device() == "cpu"


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_device_detection_user_preference_overrides(mock_mps, mock_cuda, monkeypatch):
    """Verify user preference settings override auto-detection if available."""
    mock_cuda.return_value = False
    mock_mps.return_value = True

    # Override settings
    monkeypatch.setattr(settings, "DEVICE_PREFERENCE", "cpu")
    # Even if MPS is True, settings say CPU
    assert (
        ModelRegistry.get_torch_device() == "mps"
    )  # Auto detects best as mps because cpu is default fallback not block


@patch("src.transcribe.ModelRegistry.get_whisper_model")
def test_transcribe_audio_not_found(mock_get_model):
    """Verify FileNotFoundError is raised if audio path does not exist."""
    with pytest.raises(FileNotFoundError):
        transcribe_audio(Path("non_existent.wav"))


@patch("src.transcribe.ModelRegistry.get_whisper_model")
@patch("pathlib.Path.exists")
def test_transcribe_audio_success(mock_exists, mock_get_model, tmp_path):
    """Verify that transcribing audio generates the correct files and outputs."""
    mock_exists.return_value = True

    # Mock WhisperModel segment objects
    mock_segment_1 = MagicMock()
    mock_segment_1.start = 0.5
    mock_segment_1.end = 2.5
    mock_segment_1.text = " Hello world. "
    # log prob = -0.1 -> exp(-0.1) = ~0.90 (high confidence)
    mock_segment_1.avg_logprob = -0.1
    mock_segment_1.words = [
        MagicMock(start=0.5, end=1.0, word="Hello", probability=0.95),
        MagicMock(start=1.1, end=2.5, word="world.", probability=0.88),
    ]

    mock_segment_2 = MagicMock()
    mock_segment_2.start = 3.0
    mock_segment_2.end = 5.0
    mock_segment_2.text = " Low confidence segment. "
    # log prob = -1.2 -> exp(-1.2) = ~0.30 (low confidence, below 0.6)
    mock_segment_2.avg_logprob = -1.2
    mock_segment_2.words = None

    mock_segments = [mock_segment_1, mock_segment_2]
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 6.0

    mock_model = MagicMock()
    # transcribe returns tuple (generator of segments, info)
    mock_model.transcribe.return_value = (mock_segments, mock_info)
    mock_get_model.return_value = mock_model

    input_file = Path("test_normalized.wav")
    result = transcribe_audio(input_file)

    # Check return structure
    assert result["language"] == "en"
    assert result["duration"] == 6.0
    assert len(result["segments"]) == 2

    # Check high confidence segment
    seg1 = result["segments"][0]
    assert seg1["start"] == 0.5
    assert seg1["end"] == 2.5
    assert seg1["text"] == "Hello world."
    assert seg1["confidence"] > 0.85
    assert len(seg1["words"]) == 2
    assert seg1["words"][0]["word"] == "Hello"

    # Check low confidence segment has prefix
    seg2 = result["segments"][1]
    assert seg2["text"].startswith("[Low Confidence]")

    # Check file outputs
    raw_whisper_file = settings.OUTPUT_DIR / "raw_whisper.json"
    transcript_txt_file = settings.OUTPUT_DIR / "transcript.txt"

    assert raw_whisper_file.exists()
    assert transcript_txt_file.exists()

    # Check transcript text content
    with open(transcript_txt_file, "r") as f:
        content = f.read()
    assert "[00:00:00]\nHello world." in content
    assert "[00:00:03]\n[Low Confidence] Low confidence segment." in content


@patch("src.transcribe.ModelRegistry.get_whisper_model")
@patch("pathlib.Path.exists")
def test_transcribe_audio_exception(mock_exists, mock_get_model):
    """Verify that exceptions raised by WhisperModel are wrapped in a RuntimeError."""
    mock_exists.return_value = True

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = Exception("Model out of memory")
    mock_get_model.return_value = mock_model

    with pytest.raises(RuntimeError, match="Whisper transcription failed"):
        transcribe_audio(Path("test_normalized.wav"))
