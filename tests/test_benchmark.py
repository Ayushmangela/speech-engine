import json
import pytest
from src.benchmark import get_machine_info, run_benchmark
from src.config import settings


@pytest.fixture(autouse=True)
def mock_output_dir(tmp_path, monkeypatch):
    """Overrides outputs directory setting to a temp directory during tests."""
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path / "output")


def test_get_machine_info():
    """Verify that get_machine_info extracts OS, CPU, RAM, and GPU info without error."""
    info = get_machine_info()

    assert "os" in info
    assert "cpu" in info
    assert "gpu_acceleration" in info
    assert "ram_gb" in info

    assert isinstance(info["ram_gb"], float)
    assert info["ram_gb"] >= 0.0


def test_run_benchmark():
    """Verify that run_benchmark generates a valid report JSON file with correct RTF calculation."""
    # Run with 100s audio and 10s total time -> RTF = 0.1x
    report = run_benchmark(
        audio_duration=100.0,
        transcribe_time=5.0,
        diarize_time=4.5,
        merge_time=0.5,
        total_time=10.0,
    )

    assert report["metrics"]["audio_duration_seconds"] == 100.0
    assert report["metrics"]["real_time_factor_rtf"] == 0.1
    assert report["metrics"]["total_processing_time_seconds"] == 10.0

    benchmark_file = settings.OUTPUT_DIR / "benchmark.json"
    assert benchmark_file.exists()

    with open(benchmark_file, "r") as f:
        saved_data = json.load(f)

    assert saved_data["metrics"]["real_time_factor_rtf"] == 0.1
    assert saved_data["machine_info"]["cpu"] == report["machine_info"]["cpu"]
