from typing import Any

import torch
import whisper
from whisper.decoding import DecodingOptions

from whisper_s2t.backends import WhisperModel
from whisper_s2t.configs import *

ASR_OPTIONS: dict[str, Any] = {
    "beam_size": 1,
    "without_timestamps": True,
    "return_scores": True,
    "return_no_speech_prob": True,
    "patience": 1,
    "length_penalty": 1,
}


class WhisperModelOAI(WhisperModel):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        compute_type: str = "float16",
        max_text_token_len: int = MAX_TEXT_TOKEN_LENGTH,
        asr_options: dict[str, Any] = {},
        **model_kwargs: Any,
    ) -> None:

        self.model_name: str = model_name
        self.asr_options: dict[str, Any] = ASR_OPTIONS.copy()
        self.asr_options.update(asr_options)

        self.model: Any = whisper.load_model(model_name)
        self.model.to(device).eval()

        self.decode_options: dict[str, Any] = {
            "sample_len": max_text_token_len,
            "fp16": True if compute_type == "float16" else False,
        }

        for k, v in self.asr_options.items():
            if hasattr(DecodingOptions, k):
                self.decode_options[k] = v

        super().__init__(
            device=device,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs,
        )

    def update_decode_options(self, params: dict[str, Any] = {}) -> None:
        self.decode_options.update(params)

        if "sample_len" in params:
            self.update_params(params={"max_text_token_len": params["sample_len"]})

    def generate_segment_batched(
        self,
        features: torch.Tensor,
        prompts: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:

        if self.compute_type == "float16":
            features = features.to(self.device).half()

        lang_and_task_pairs: dict[tuple[Any, Any], list[int]] = {}
        for _i, _p in enumerate(prompts):
            try:
                lang_and_task_pairs[(_p[-3], _p[-2])].append(_i)
            except KeyError:
                lang_and_task_pairs[(_p[-3], _p[-2])] = [_i]

        response: list[dict[str, Any]] = [{} for _ in prompts]
        for (task, lang), idx_list in lang_and_task_pairs.items():

            results = self.model.decode(
                features[idx_list].to(self.device),
                DecodingOptions(task=task, language=lang, **self.decode_options),
            )

            for idx, result in zip(idx_list, results):
                response[idx]["text"] = result.text.strip()

                if self.asr_options["return_scores"]:
                    response[idx]["avg_logprob"] = result.avg_logprob

                if self.asr_options["return_no_speech_prob"]:
                    response[idx]["no_speech_prob"] = result.no_speech_prob

        return response
