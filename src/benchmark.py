import json
import os
import platform
import subprocess
from typing import Dict, Any
import torch
from src.config import settings
from src.logger import logger


def get_machine_info() -> Dict[str, Any]:
    """
    Retrieves hardware and operating system specifications for the host machine.
    Uses POSIX sysconf for RAM and sysctl for macOS CPU.

    Returns:
        Dict[str, Any]: Machine information dictionary.
    """
    os_name = platform.system()
    os_version = platform.release()

    # Get CPU info
    if os_name == "Darwin":
        try:
            cpu_brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            )
            cpu_info = cpu_brand.stdout.strip()
        except Exception:
            cpu_info = platform.processor() or "Apple Silicon"
    else:
        cpu_info = platform.processor() or "Unknown CPU"

    # Get RAM info in gigabytes using POSIX sysconf
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        num_pages = os.sysconf("SC_PHYS_PAGES")
        total_ram_gb = (page_size * num_pages) / (1024**3)
    except Exception:
        total_ram_gb = 0.0

    # Get GPU/MPS/CUDA info
    device_info = "CPU"
    if torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device_info = "Apple Metal (MPS)"

    return {
        "os": f"{os_name} {os_version}",
        "cpu": cpu_info,
        "gpu_acceleration": device_info,
        "ram_gb": round(total_ram_gb, 2),
    }


def run_benchmark(
    audio_duration: float,
    transcribe_time: float,
    diarize_time: float,
    merge_time: float,
    total_time: float,
) -> Dict[str, Any]:
    """
    Collects performance metrics and saves a benchmark report to output/benchmark.json.
    Logs a console summary.

    Args:
        audio_duration: Audio length in seconds.
        transcribe_time: Transcription duration in seconds.
        diarize_time: Diarization duration in seconds.
        merge_time: Alignment merge duration in seconds.
        total_time: End-to-end processing duration in seconds.

    Returns:
        Dict[str, Any]: Generated benchmark report dictionary.
    """
    logger.info("Generating performance benchmark report...")

    machine_info = get_machine_info()
    rtf = total_time / audio_duration if audio_duration > 0 else 0.0

    report = {
        "machine_info": machine_info,
        "configuration": {
            "whisper_model": settings.WHISPER_MODEL,
            "device_preference": settings.DEVICE_PREFERENCE,
        },
        "metrics": {
            "audio_duration_seconds": round(audio_duration, 2),
            "transcription_time_seconds": round(transcribe_time, 2),
            "diarization_time_seconds": round(diarize_time, 2),
            "merge_time_seconds": round(merge_time, 2),
            "total_processing_time_seconds": round(total_time, 2),
            "real_time_factor_rtf": round(rtf, 3),
        },
    }

    # Save output/benchmark.json
    try:
        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        benchmark_file = settings.OUTPUT_DIR / "benchmark.json"
        with open(benchmark_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Benchmark report saved to {benchmark_file}")
    except Exception as e:
        logger.exception("Failed to save benchmark.json")
        raise RuntimeError(f"Could not save benchmark.json: {e}") from e

    # Log summary formatted precisely like the user's example
    duration_str = f"{int(audio_duration // 60)}m{int(audio_duration % 60)}s"
    logger.info("--- PERFORMANCE BENCHMARK ---")
    logger.info(f"Audio Length  : {duration_str}")
    logger.info(f"Transcription : {transcribe_time:.1f}s")
    logger.info(f"Diarization   : {diarize_time:.1f}s")
    logger.info(f"Merge         : {merge_time:.1f}s")
    logger.info(f"Total Time    : {total_time:.1f}s")
    logger.info(f"RTF           : {rtf:.2f}x")
    logger.info("-----------------------------")

    return report
