import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.main import app
from src.config import settings

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_output_dir(tmp_path, monkeypatch):
    """Overrides RAW_AUDIO_DIR during tests."""
    monkeypatch.setattr(settings, "RAW_AUDIO_DIR", tmp_path / "raw")


@patch("src.main.run_pipeline")
def test_api_diarize_success(mock_run_pipeline):
    """Verify that uploading a file returns the correct JSON response structure and status code."""
    mock_run_pipeline.return_value = [
        {
            "speaker": "Speaker 1",
            "start": 0.20,
            "end": 2.15,
            "text": "Hello everyone.",
            "confidence": 0.95,
        },
        {
            "speaker": "Speaker 2",
            "start": 2.50,
            "end": 4.10,
            "text": "Hi.",
            "confidence": 0.88,
        },
    ]

    files = {"file": ("interview.wav", b"fake_wav_audio_content", "audio/wav")}
    response = client.post("/api/v1/diarize", files=files)

    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2
    assert data[0]["speaker"] == "Speaker 1"
    assert data[0]["start"] == 0.20
    assert data[0]["end"] == 2.15
    assert data[0]["text"] == "Hello everyone."
    assert data[0]["confidence"] == 0.95

    assert data[1]["speaker"] == "Speaker 2"
    assert data[1]["text"] == "Hi."

    mock_run_pipeline.assert_called_once()


@patch("src.main.run_pipeline")
def test_api_diarize_server_error(mock_run_pipeline):
    """Verify that internal pipeline failures are wrapped in a 500 error response."""
    mock_run_pipeline.side_effect = RuntimeError("FFmpeg not found")

    files = {"file": ("interview.wav", b"fake_wav_audio_content", "audio/wav")}
    response = client.post("/api/v1/diarize", files=files)

    assert response.status_code == 500
    assert "Audio processing failed" in response.json()["detail"]
