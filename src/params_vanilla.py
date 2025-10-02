from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HFTrainingArguments
# from trl import DPOConfig as DPOConfigTRL
from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    loss_lvr_fct: str = field(default="mse")
    loss_lvr_lambda: float = field(default=1e-1)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lvr_head_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    # use_liger: bool = True
    run_name: Optional[str] = field(default="vscode debugger", metadata={"help": "Name of the run for logging purposes."})
    # True if serving the checkpoints and data on oci
    online_checkpoint: Optional[bool] = False
    checkpoint_name:Optional[str] = None
    # data packing-related params
    enable_data_packing: bool = False
    max_packed_tokens:Optional[int] = None
    # long_seq_cut:Optional[int] = field(default=25600, metadata={"help": "Max Len of long single data instnace allowed"})
    long_seq_threshold:Optional[int] = field(default=4096, metadata={"help": "Threshold to be a long single instance"})
    max_instance_per_batch:Optional[int] = 4

    mode_switch_loss: Optional[bool] = False
    loss_mode_switch_fct: Optional[str] = field(default="mse")
    loss_mode_switch_lambda:Optional[float] = field(default=1e-1)


@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = 256
    max_prompt_length: int = 512

    online_checkpoint: Optional[bool] = False
    checkpoint_name:Optional[str] = None
    decoding_strategy:str = "steps"
    lvr_steps: int = 16



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0
    random_seed: Optional[int] = field(default=None)
    