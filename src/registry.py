import torch
from typing import Optional, Any
from faster_whisper import WhisperModel
from src.config import settings
from src.logger import logger


class ModelRegistry:
    """Singleton registry to cache and manage AI models (Whisper, pyannote) across requests."""

    _whisper_model: Optional[WhisperModel] = None
    _diarization_pipeline: Optional[Any] = None

    @classmethod
    def get_torch_device(cls) -> str:
        """
        Detects the best available torch device based on preference and hardware capability.
        Priority: CUDA -> MPS -> CPU.
        """
        pref = settings.DEVICE_PREFERENCE.lower()

        # Check user preference first
        if pref == "cuda" and torch.cuda.is_available():
            logger.info("Using CUDA device based on preference configuration.")
            return "cuda"
        if pref == "mps" and torch.backends.mps.is_available():
            logger.info(
                "Using MPS (Apple Metal) device based on preference configuration."
            )
            return "mps"

        # Fallback to auto-detection
        if torch.cuda.is_available():
            logger.info("Detected CUDA device available. Auto-selecting cuda.")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info(
                "Detected MPS (Apple Metal) device available. Auto-selecting mps."
            )
            return "mps"

        logger.info("No GPU acceleration found. Defaulting to CPU.")
        return "cpu"

    @classmethod
    def get_whisper_model(cls) -> WhisperModel:
        """
        Loads and caches the faster-whisper model.
        CTranslate2 only supports CPU and CUDA. If device is MPS, fallback to CPU.
        """
        if cls._whisper_model is None:
            model_size = settings.WHISPER_MODEL
            torch_device = cls.get_torch_device()
            whisper_device = "cuda" if torch_device == "cuda" else "cpu"
            compute_type = "float16" if whisper_device == "cuda" else "int8"

            logger.info(
                f"Loading faster-whisper model '{model_size}' on device '{whisper_device}' "
                f"(CTranslate2 fallback from torch device '{torch_device}', compute_type='{compute_type}')..."
            )

            try:
                cls._whisper_model = WhisperModel(
                    model_size, device=whisper_device, compute_type=compute_type
                )
                logger.info(f"faster-whisper model '{model_size}' loaded successfully.")
            except Exception as e:
                logger.exception(f"Failed to load Whisper model '{model_size}'")
                raise RuntimeError(
                    f"Failed to load Whisper model '{model_size}': {e}"
                ) from e

        return cls._whisper_model

    @classmethod
    def get_diarization_pipeline(cls) -> Any:
        """
        Loads and caches the pyannote.audio diarization pipeline.
        Uses HF_TOKEN environment variable.
        """
        if cls._diarization_pipeline is None:
            # We import here to avoid loading pyannote.audio if not needed
            from pyannote.audio import Pipeline

            hf_token = settings.HF_TOKEN
            if not hf_token or hf_token == "your_huggingface_token_here":
                logger.warning(
                    "HF_TOKEN is not configured or is a placeholder. pyannote.audio loading might fail."
                )

            logger.info(
                "Loading pyannote/speaker-diarization-3.1 pipeline from Hugging Face..."
            )
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
                )
                if pipeline is None:
                    raise RuntimeError(
                        "Pipeline returned None. Check HF model access terms or HF_TOKEN validity."
                    )

                torch_device = cls.get_torch_device()
                logger.info(f"Sending pyannote pipeline to device '{torch_device}'")
                pipeline.to(torch.device(torch_device))
                cls._diarization_pipeline = pipeline
                logger.info(
                    "pyannote diarization pipeline loaded and configured successfully."
                )
            except Exception as e:
                logger.exception("Failed to load pyannote speaker diarization pipeline")
                raise RuntimeError(
                    f"Failed to load pyannote speaker diarization pipeline: {e}"
                ) from e

        return cls._diarization_pipeline
