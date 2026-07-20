from unittest.mock import patch, MagicMock
import pytest
import torch
from src.registry import ModelRegistry


from src.config import settings


@pytest.fixture(autouse=True)
def reset_registry():
    """Resets ModelRegistry internal state before each test."""
    ModelRegistry._whisper_model = None
    ModelRegistry._diarization_pipeline = None


@patch("src.registry.WhisperModel")
@patch("src.registry.ModelRegistry.get_torch_device")
def test_whisper_model_caching(mock_get_device, mock_whisper_class):
    """Verify that faster-whisper model is loaded only once and cached."""
    mock_get_device.return_value = "cpu"
    mock_instance = MagicMock()
    mock_whisper_class.return_value = mock_instance

    model1 = ModelRegistry.get_whisper_model()
    model2 = ModelRegistry.get_whisper_model()

    assert model1 is model2
    mock_whisper_class.assert_called_once_with(
        settings.WHISPER_MODEL, device="cpu", compute_type="int8"
    )


@patch("pyannote.audio.Pipeline.from_pretrained")
@patch("src.registry.ModelRegistry.get_torch_device")
def test_pyannote_pipeline_caching(mock_get_device, mock_from_pretrained):
    """Verify that pyannote pipeline is loaded only once and cached on the correct device."""
    mock_get_device.return_value = "cpu"
    mock_pipeline = MagicMock()
    mock_from_pretrained.return_value = mock_pipeline

    pipeline1 = ModelRegistry.get_diarization_pipeline()
    pipeline2 = ModelRegistry.get_diarization_pipeline()

    assert pipeline1 is pipeline2
    mock_from_pretrained.assert_called_once()
    mock_pipeline.to.assert_called_once_with(torch.device("cpu"))


@patch("pyannote.audio.Pipeline.from_pretrained")
def test_pyannote_pipeline_returns_none(mock_from_pretrained):
    """Verify exception wrapping when the pyannote pipeline fails to load."""
    mock_from_pretrained.return_value = None

    with pytest.raises(RuntimeError, match="Pipeline returned None"):
        ModelRegistry.get_diarization_pipeline()


@patch("src.registry.WhisperModel")
def test_whisper_model_load_error(mock_whisper_class):
    """Verify exception wrapping when faster-whisper fails to instantiate."""
    mock_whisper_class.side_effect = ValueError("Invalid model path")

    with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
        ModelRegistry.get_whisper_model()
