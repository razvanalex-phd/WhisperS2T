import hashlib
import json
import os

import pynvml
from rich.console import Console

console = Console()

from ....utils import RunningStatus
from .download_utils import SAVE_DIR, download_model


class TRTBuilderConfig:
    def __init__(
        self,
        max_batch_size=24,
        max_beam_width=1,
        max_input_len_enc=3000,
        max_input_len_dec=14,
        max_output_len=512,
        world_size=1,
        dtype="float16",
        quantize_dir="quantize/1-gpu",
        use_gpt_attention_plugin="float16",
        use_bert_attention_plugin=None,
        use_context_fmha_enc=False,
        use_context_fmha_dec=False,
        use_gemm_plugin="float16",
        remove_input_padding=True,
        use_weight_only=True,
        weight_only_precision="int8",
        int8_kv_cache=False,
        debug_mode=False,
        logits_dtype="float16",
        **kwargs,
    ):

        self.max_batch_size = max_batch_size
        self.max_beam_width = max_beam_width
        self.max_input_len_enc = max_input_len_enc
        self.max_input_len_dec = max_input_len_dec
        self.max_output_len = max_output_len
        self.world_size = world_size
        self.dtype = dtype
        self.quantize_dir = quantize_dir
        self.use_gpt_attention_plugin = use_gpt_attention_plugin
        self.use_bert_attention_plugin = use_bert_attention_plugin
        self.use_context_fmha_enc = use_context_fmha_enc
        self.use_context_fmha_dec = use_context_fmha_dec
        self.use_gemm_plugin = use_gemm_plugin
        self.remove_input_padding = remove_input_padding
        self.use_weight_only = use_weight_only
        self.weight_only_precision = weight_only_precision
        self.int8_kv_cache = int8_kv_cache
        self.debug_mode = debug_mode
        self.logits_dtype = logits_dtype
        self.output_dir = None
        self.model_dir = None
        self.checkpoint_dir = None
        self.model_name = None

        pynvml.nvmlInit()
        self.cuda_compute_capability = list(
            pynvml.nvmlDeviceGetCudaComputeCapability(
                pynvml.nvmlDeviceGetHandleByIndex(0)
            )
        )
        pynvml.nvmlShutdown()

    def identifier(self):
        params = vars(self)
        return hashlib.md5(json.dumps(params).encode()).hexdigest()


def save_trt_build_configs(trt_build_args):
    with open(f"{trt_build_args.output_dir}/trt_build_args.json", "w") as f:
        f.write(json.dumps(vars(trt_build_args)))


def load_trt_build_config(output_dir):
    """
    [TODO]: Add cuda_compute_capability verification check
    """

    with open(f"{output_dir}/trt_build_args.json", "r") as f:
        trt_build_configs = json.load(f)

    trt_build_args = TRTBuilderConfig(**trt_build_configs)
    trt_build_args.output_dir = trt_build_configs["output_dir"]
    trt_build_args.model_dir = trt_build_configs["model_dir"]
    trt_build_args.checkpoint_dir = trt_build_args.output_dir + "_weights"
    trt_build_args.model_name = trt_build_configs["model_name"]

    return trt_build_args


def build_trt_engine(model_name="large-v2", args=None, force=False, log_level="error"):

    if args is None:
        console.print(f"args is None, using default configs.")
        args = TRTBuilderConfig()

    # TODO: revert the following
    args.output_dir = os.path.join(SAVE_DIR, model_name, args.identifier())  # type: ignore
    model_path, tokenizer_path = download_model(model_name)
    args.model_dir = os.path.dirname(model_path)  # type: ignore
    args.checkpoint_dir = args.output_dir + "_weights"  # type: ignore
    args.model_name = model_name  # type: ignore

    if force:
        console.print(f"'force' flag is 'True'. Removing previous build.")
        with RunningStatus("Cleaning", console=console):
            os.system(f"rm -rf '{args.output_dir}'")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        _files = [
            os.path.join(os.path.relpath(dp, args.output_dir), f)
            for dp, dn, fn in os.walk(args.output_dir)
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
        "Exporting Model To TensorRT Engine (3-6 mins)", console=console
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
