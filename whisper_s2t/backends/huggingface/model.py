from typing import Any, cast

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_s2t.backends import WhisperModel
from whisper_s2t.configs import *

ASR_OPTIONS = {
    "beam_size": 1,
    "without_timestamps": True,
    "return_scores": False,
    "return_no_speech_prob": False,
    "use_flash_attention": True,
}


COMPUTE_TYPE_TO_TORCH_DTYPE = {"float16": torch.float16}


class WhisperModelHF(WhisperModel):
    def __init__(
        self,
        model_name: str,
        device="cuda",
        compute_type="float16",
        max_text_token_len=MAX_TEXT_TOKEN_LENGTH,
        asr_options={},
        **model_kwargs,
    ) -> None:
        self.model_name = model_name
        self.asr_options = ASR_OPTIONS
        self.asr_options.update(asr_options)

        self.processor = cast(
            WhisperProcessor,
            WhisperProcessor.from_pretrained(self.model_name),
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=COMPUTE_TYPE_TO_TORCH_DTYPE.get(compute_type, torch.float32),
            low_cpu_mem_usage=True,
            use_safetensors=True,
            use_flash_attention_2=self.asr_options["use_flash_attention"],
        )
        self.model.config.forced_decoder_ids = None
        self.model.to(device).eval()  # type: ignore

        self.generate_kwargs = {
            "max_new_tokens": max_text_token_len,
            "num_beams": self.asr_options["beam_size"],
            "return_timestamps": not self.asr_options["without_timestamps"],
        }

        super().__init__(
            device=device,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs,
        )

    def update_generation_kwargs(self, params={}):
        self.generate_kwargs.update(params)

        if "max_new_tokens" in params:
            self.update_params(params={"max_text_token_len": params["max_new_tokens"]})

    def generate_segment_batched(
        self,
        features: torch.Tensor,
        prompts: list[list[int]],
        seq_lens: torch.Tensor,
        seg_metadata: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self.compute_type == "float16":
            features = features.to(self.device).half()

        attention_mask = torch.ones(
            features.shape[0], features.shape[1], device=features.device
        )
        for i, length in enumerate(seq_lens):
            if length < features.shape[1]:
                attention_mask[i, length:] = 0

        lang_and_task_pairs = {}
        for _i, _p in enumerate(prompts):
            try:
                lang_and_task_pairs[(_p[-3], _p[-2])].append(_i)
            except:
                lang_and_task_pairs[(_p[-3], _p[-2])] = [_i]

        response = [{} for _ in prompts]
        for (task, lang), idx_list in lang_and_task_pairs.items():
            batch_generate_kwargs = self.generate_kwargs.copy()
            batch_generate_kwargs["forced_decoder_ids"] = (
                self.processor.get_decoder_prompt_ids(task=task, language=lang)
            )

            predicted_ids = self.model.generate(
                features[idx_list],
                attention_mask=attention_mask[idx_list],
                **batch_generate_kwargs,
            )

            results = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )

            for idx, text in zip(idx_list, results):
                response[idx]["text"] = text.strip()

        return response
