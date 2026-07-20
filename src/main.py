import argparse
import os
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import logger
from src.normalize import validate_audio, normalize_audio
from src.transcribe import transcribe_audio
from src.diarize import diarize_audio
from src.merge import merge_transcript_and_diarization
from src.benchmark import run_benchmark

app = FastAPI(
    title="Speech-to-Text and Diarization API",
    description="A local pipeline combining FFmpeg normalization, faster-whisper, and pyannote.audio speaker diarization.",
    version="1.0.0",
)

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MergedSegment(BaseModel):
    """Output schema representing a transcribed text segment matched to a speaker."""

    speaker: str = Field(..., description="Sequential speaker label (e.g. Speaker 1)")
    start: float = Field(..., description="Segment start timestamp in seconds")
    end: float = Field(..., description="Segment end timestamp in seconds")
    text: str = Field(..., description="Transcribed segment text")
    confidence: float = Field(
        ..., description="Confidence score from Whisper [0.0 - 1.0]"
    )


def run_pipeline(audio_path: Path) -> List[Dict[str, Any]]:
    """
    Executes the full local audio transcription and speaker diarization pipeline.
    This function is CLI and FastAPI independent.

    Args:
        audio_path: Path to the raw input audio file.

    Returns:
        List[Dict[str, Any]]: List of merged speaker segments.
    """
    logger.info(f"--- STARTING PIPELINE RUN FOR {audio_path.name} ---")
    start_time = time.time()

    # Ensure necessary folders exist
    settings.RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    settings.NORMALIZED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Validation & Metadata Capture (generates output/audio_metadata.json)
    metadata_start = time.time()
    metadata = validate_audio(audio_path)
    metadata_duration = time.time() - metadata_start
    logger.info(f"Audio validation completed in {metadata_duration:.2f}s")

    # Copy raw audio into RAW_AUDIO_DIR if not already there
    raw_saved_path = settings.RAW_AUDIO_DIR / audio_path.name
    if audio_path.resolve() != raw_saved_path.resolve():
        try:
            shutil.copy2(audio_path, raw_saved_path)
            logger.info(f"Raw audio saved to {raw_saved_path}")
            active_audio_path = raw_saved_path
        except Exception as e:
            logger.warning(
                f"Could not copy raw audio: {e}. Processing original file directly."
            )
            active_audio_path = audio_path
    else:
        active_audio_path = audio_path

    # 2. Normalization (Mono, 16kHz, WAV PCM)
    norm_start = time.time()
    normalized_path = normalize_audio(active_audio_path)
    norm_duration = time.time() - norm_start
    logger.info(f"Audio normalization completed in {norm_duration:.2f}s")

    # 3. Transcription (generates output/raw_whisper.json)
    transcribe_start = time.time()
    whisper_result = transcribe_audio(normalized_path)
    transcribe_duration = time.time() - transcribe_start
    logger.info(f"Transcription completed in {transcribe_duration:.2f}s")

    # 4. Diarization (generates output/raw_diarization.json)
    diarize_start = time.time()
    diarization_result = diarize_audio(normalized_path)
    diarize_duration = time.time() - diarize_start
    logger.info(f"Diarization completed in {diarize_duration:.2f}s")

    # 5. Merge Timeline (generates output/merged.json and output/transcript.txt)
    merge_start = time.time()
    merged_result = merge_transcript_and_diarization(whisper_result, diarization_result)
    merge_duration = time.time() - merge_start
    logger.info(f"Merging completed in {merge_duration:.2f}s")

    # 6. Performance Benchmarking (generates output/benchmark.json)
    total_time = time.time() - start_time
    audio_duration = metadata.get("duration", 0.0)
    run_benchmark(
        audio_duration=audio_duration,
        transcribe_time=transcribe_duration,
        diarize_time=diarize_duration,
        merge_time=merge_duration,
        total_time=total_time,
    )

    logger.info(f"--- PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f}s ---")
    return merged_result


@app.post(
    "/api/v1/diarize",
    response_model=List[MergedSegment],
    status_code=status.HTTP_200_OK,
    summary="Diarize and transcribe an uploaded audio file",
)
async def api_diarize(
    file: UploadFile = File(..., description="Audio file to diarize and transcribe")
):
    """
    Uploads an audio file and executes the pipeline synchronously,
    returning the merged speaker transcript as JSON.
    """
    logger.info(f"Received API upload request for file: {file.filename}")

    # Ensure temporary raw directory exists
    settings.RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Create safe temporary file path in RAW_AUDIO_DIR
    temp_path = settings.RAW_AUDIO_DIR / f"upload_{time.time()}_{file.filename}"

    try:
        # Save file to disk
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Uploaded file saved temporarily to: {temp_path}")

        # Execute pipeline
        merged_result = run_pipeline(temp_path)
        return merged_result

    except Exception as e:
        logger.exception("Exception occurred during API request processing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {str(e)}",
        )
    finally:
        # Clean up temporary upload file if it exists
        if temp_path.exists():
            try:
                os.remove(temp_path)
                logger.info(f"Temporary upload file removed: {temp_path}")
            except Exception as cleanup_err:
                logger.warning(
                    f"Could not clean up temporary file {temp_path}: {cleanup_err}"
                )


def cli() -> None:
    """CLI entrypoint for executing the pipeline from the terminal."""
    parser = argparse.ArgumentParser(
        description="Speech-to-Text and Diarization Local Pipeline CLI"
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to raw audio file to process"
    )
    args = parser.parse_args()

    audio_file_path = Path(args.audio)
    try:
        result = run_pipeline(audio_file_path)
        print("\nPipeline execution complete! Merged results summary:")
        from src.transcribe import format_timestamp

        for turn in result:
            print(
                f"[{format_timestamp(turn['start'])}] {turn['speaker']}: {turn['text']}"
            )
    except Exception as err:
        print(f"\nPipeline execution failed: {err}")
        import sys

        sys.exit(1)


if __name__ == "__main__":
    cli()
