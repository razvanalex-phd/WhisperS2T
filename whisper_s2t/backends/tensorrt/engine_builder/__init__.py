import hashlib
import json
import os
from typing import Any

import pynvml
from rich.console import Console

console = Console()

from whisper_s2t.backends.tensorrt.engine_builder.download_utils import (
    SAVE_DIR,
    download_model,
)
from whisper_s2t.utils import RunningStatus


class TRTBuilderConfig:
    def __init__(
        self,
        max_batch_size: int = 24,
        max_beam_width: int = 1,
        max_input_len_enc: int = 3000,
        max_input_len_dec: int = 14,
        max_output_len: int = 512,
        world_size: int = 1,
        dtype: str = "float16",
        quantize_dir: str = "quantize/1-gpu",
        use_gpt_attention_plugin: str | None = "float16",
        use_bert_attention_plugin: str | None = None,
        use_context_fmha_enc: bool = False,
        use_context_fmha_dec: bool = False,
        use_gemm_plugin: str | None = "float16",
        remove_input_padding: bool = True,
        use_weight_only: bool = True,
        weight_only_precision: str = "int8",
        int8_kv_cache: bool = False,
        debug_mode: bool = False,
        logits_dtype: str = "float16",
        **kwargs: Any,
    ) -> None:
        self.max_batch_size: int = max_batch_size
        self.max_beam_width: int = max_beam_width
        self.max_input_len_enc: int = max_input_len_enc
        self.max_input_len_dec: int = max_input_len_dec
        self.max_output_len: int = max_output_len
        self.world_size: int = world_size
        self.dtype: str = dtype
        self.quantize_dir: str = quantize_dir
        self.use_gpt_attention_plugin: str | None = use_gpt_attention_plugin
        self.use_bert_attention_plugin: str | None = use_bert_attention_plugin
        self.use_context_fmha_enc: bool = use_context_fmha_enc
        self.use_context_fmha_dec: bool = use_context_fmha_dec
        self.use_gemm_plugin: str | None = use_gemm_plugin
        self.remove_input_padding: bool = remove_input_padding
        self.use_weight_only: bool = use_weight_only
        self.weight_only_precision: str = weight_only_precision
        self.int8_kv_cache: bool = int8_kv_cache
        self.debug_mode: bool = debug_mode
        self.logits_dtype: str = logits_dtype
        self.output_dir: str | None = None
        self.model_dir: str | None = None
        self.checkpoint_dir: str | None = None
        self.model_name: str | None = None

        pynvml.nvmlInit()
        self.cuda_compute_capability: list[int] = list(
            pynvml.nvmlDeviceGetCudaComputeCapability(
                pynvml.nvmlDeviceGetHandleByIndex(0)
            )
        )
        pynvml.nvmlShutdown()

    def identifier(self) -> str:
        params = vars(self)
        return hashlib.md5(json.dumps(params).encode()).hexdigest()


def save_trt_build_configs(trt_build_args: TRTBuilderConfig) -> None:
    with open(f"{trt_build_args.output_dir}/trt_build_args.json", "w") as f:
        f.write(json.dumps(vars(trt_build_args)))


def load_trt_build_config(output_dir: str) -> TRTBuilderConfig:
    """
    [TODO]: Add cuda_compute_capability verification check
    """

    with open(f"{output_dir}/trt_build_args.json", "r") as f:
        trt_build_configs = json.load(f)

    trt_build_args = TRTBuilderConfig(**trt_build_configs)

    trt_build_args.output_dir = trt_build_configs["output_dir"]
    assert trt_build_args.output_dir and os.path.exists(
        trt_build_args.output_dir
    ), f"Output directory does not exist: {trt_build_args.output_dir}"

    trt_build_args.model_dir = trt_build_configs["model_dir"]
    trt_build_args.checkpoint_dir = trt_build_args.output_dir + "_weights"
    trt_build_args.model_name = trt_build_configs["model_name"]

    return trt_build_args


def build_trt_engine(
    model_name: str = "large-v2",
    args: TRTBuilderConfig | None = None,
    force: bool = False,
    _log_level: str = "error",
) -> str:
    if args is None:
        console.print(f"args is None, using default configs.")
        args = TRTBuilderConfig()

    # TODO: revert the following
    args.output_dir = os.path.join(SAVE_DIR, model_name, args.identifier())
    model_path, tokenizer_path = download_model(model_name)
    args.model_dir = os.path.dirname(model_path)
    args.checkpoint_dir = args.output_dir + "_weights"
    args.model_name = model_name

    if force:
        console.print(f"'force' flag is 'True'. Removing previous build.")
        with RunningStatus("Cleaning", console=console):
            os.system(f"rm -rf '{args.output_dir}'")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        _files = [
            os.path.join(os.path.relpath(dp, args.output_dir), f)
            for dp, _dn, fn in os.walk(args.output_dir)
            for f in fn
        ]

        _failed_export = False
        for _req_files in [
            "./tokenizer.json",
            "./trt_build_args.json",
            "encoder/config.json",
            "encoder/rank0.engine",
            "decoder/config.json",
            "decoder/rank0.engine",
        ]:
            if _req_files not in _files:
                _failed_export = True
                break

        if _failed_export:
            console.print(
                f"Export directory exists but seems like a failed export, regenerating the engine files."
            )
            os.system(f"rm -rf '{args.output_dir}'")
            os.makedirs(args.output_dir)
        else:
            return args.output_dir

    os.system(f"cp '{tokenizer_path}' '{args.output_dir}/tokenizer.json'")
    save_trt_build_configs(args)

    with RunningStatus(
        "Exporting Model To TensorRT Engine (3-6 mins)",
        console=console,
    ):
        cmd = f"python3 -m whisper_s2t.backends.tensorrt.engine_builder.builder --output_dir='{args.output_dir}'"
        out_logs = os.popen(cmd).read().split("\n")
        print_flag = False
        for line in out_logs:
            if print_flag:
                console.print(line)
            elif "TRTBuilderConfig" in line:
                print_flag = True
                console.print("[TRTBuilderConfig]:")

    return args.output_dir
