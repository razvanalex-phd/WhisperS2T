import os
from typing import Any, Literal

from platformdirs import user_cache_dir

BASE_PATH = os.path.dirname(__file__)

CACHE_DIR = user_cache_dir("whisper_s2t")
os.makedirs(CACHE_DIR, exist_ok=True)

Backend = Literal[
    "CTranslate2",
    "ct2",
    "huggingface",
    "hf",
    "openai",
    "oai",
    "tensorrt",
    "trt",
    "trt-llm",
    "tensorrt-llm",
    "trt_llm",
    "tensorrt_llm",
]

ModelIdentifier = Literal[
    "large-v2",
    "large-v3",
    "distil-large-v2",
]


def load_model(
    model_identifier: ModelIdentifier = "large-v2",
    backend: Backend = "CTranslate2",
    **kwargs: dict[str, Any],
):
    model_kwargs = dict(**kwargs)
    identifier = model_identifier.lower()

    if identifier in ["large-v3"]:
        model_kwargs["n_mels"] = 128
    elif (identifier in ["distil-large-v2"]) and (
        backend.lower() not in ["huggingface", "hf"]
    ):
        print(
            f"Switching backend to HuggingFace. Distill whisper is only supported with HuggingFace backend."
        )
        backend = "huggingface"

        model_kwargs["max_speech_len"] = 15.0
        model_kwargs["max_text_token_len"] = 128

    if backend.lower() in ["ctranslate2", "ct2"]:
        from .backends.ctranslate2.model import WhisperModelCT2 as WhisperModel

    elif backend.lower() in ["huggingface", "hf"]:
        from .backends.huggingface.model import WhisperModelHF as WhisperModel

        if "distil" in identifier:
            identifier = f"distil-whisper/{identifier}"
        else:
            identifier = f"openai/whisper-{identifier}"

    elif backend.lower() in ["openai", "oai"]:
        from .backends.openai.model import WhisperModelOAI as WhisperModel

    elif backend.lower() in [
        "tensorrt",
        "trt",
        "trt-llm",
        "tensorrt-llm",
        "trt_llm",
        "tensorrt_llm",
    ]:
        from .backends.tensorrt.model import WhisperModelTRT as WhisperModel
    else:
        raise ValueError(
            f"Backend name '{backend}' is invalid. Only following options are available: ['CTranslate2', 'TensorRT-LLM', 'HuggingFace', 'OpenAI']"
        )

    return WhisperModel(identifier, **model_kwargs)
