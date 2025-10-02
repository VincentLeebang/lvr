import os
import sys
sys.path.append("/dockerx/bangzhli/projects/LVR-Finetune")




import torch
import torch.nn.functional as F
from typing import Optional, Tuple

import warnings
import torch
from torch import nn
import datasets
from typing import Union
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union
from datasets import Dataset, IterableDataset
from packaging import version
import transformers
import textwrap

from torch.utils.data import DataLoader, Sampler

from accelerate.utils import is_peft_model, set_seed, broadcast_object_list, gather, gather_object
from transformers.utils import is_datasets_available

from transformers.trainer import (
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
)

from transformers.trainer import (
    ExportableState,
    SaveStrategy
)

from transformers.trainer_utils import seed_worker
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForSequenceClassification,
    GenerationConfig,
    PreTrainedModel,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from trl.trainer.utils import selective_log_softmax
from trl import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.import_utils import is_deepspeed_available, is_liger_kernel_available, is_rich_available
from trl.trainer.callbacks import SyncRefModelCallback

from trl.extras.profiling import profiling_decorator, profiling_context
from trl.data_utils import maybe_apply_chat_template, is_conversational, apply_chat_template
from trl.trainer.utils import (
    pad,
    generate_model_card,
    print_prompt_completions_sample,
    get_comet_experiment_url,
)

from src.train.train_utils import get_peft_state_non_lora_maybe_zero_3
from src.constants import MULTIMODAL_KEYWORDS
from src.model.qwen_lvr_model import QwenWithLVR
from transformers import AutoConfig
from qwen_vl_utils import process_vision_info

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class RepeatRandomSampler(RepeatSampler):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "RepeatRandomSampler is deprecated and will be removed in version 0.18. Use RepeatSampler instead.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size] if tensor is not None else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class QwenGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model_pth: Optional[str],   # The checkpoint of ref_model
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        temp_folder=None,
        oci_handler=None,
    ):
        if oci_handler:
            self.oci_handler = oci_handler
            self.temp_folder = temp_folder     # temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"

        
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}

        model_id = model.config._name_or_path
        if args.model_init_kwargs is not None:
            raise ValueError(
                "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                "This argument can only be used when the `model` argument is a string."
            )

        # Enable gradient checkpointing if requested
        # if args.gradient_checkpointing:
        #     model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        self.ref_model_pth = ref_model_pth
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None    
        elif is_deepspeed_zero3_enabled():
            if "Qwen2.5" in model_id:
                # 1. forward func already patched
                # 2. config
                ref_model_config = AutoConfig.from_pretrained(self.ref_model_pth,trust_remote_code=True)
                compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
                self.ref_model = QwenWithLVR.from_pretrained(
                    self.ref_model_pth,
                    config=ref_model_config,
                    torch_dtype=compute_dtype,
                    attn_implementation="flash_attention_2" if not args.disable_flash_attn2 else "sdpa",
                )
            else:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    **model_init_kwargs,
                )
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            raise "Please pass the processor to grapo trainer"

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(reward_funcs[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features
        
        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )
        
        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.gradient_accumulation_steps
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # Check if the effective batch size can be divided by the number of generations
        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )
        num_processes = self.accelerator.num_processes
        effective_batch_size = args.per_device_train_batch_size * num_processes * args.gradient_accumulation_steps
        possible_values = [
            n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0
        ]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The effective train batch size ({num_processes} x {args.per_device_train_batch_size} x "
                f"{args.gradient_accumulation_steps}) must be evenly divisible by the number of generations per "
                f"prompt ({self.num_generations}). Given the current effective train batch size, the valid values for "
                f"the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            effective_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [
                n_gen for n_gen in range(2, effective_batch_size + 1) if (effective_batch_size) % n_gen == 0
            ]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The effective eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be "
                    f"evenly divisible by the number of generations per prompt ({self.num_generations}). Given the "
                    "current effective eval batch size, the valid values for the number of generations are: "
                    f"{possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.tokenizer.pad_token_id,
                eos_token_id=processing_class.tokenizer.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
                decoding_strategy=args.decoding_strategy,
                lvr_steps=args.lvr_steps
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch, our dataloader loads an *accumulated* batch
    # (i.e., `per_device_batch_size × gradient_accumulation_steps`). This allows us to generate completions
    # once per optimization step—rather than once per gradient accumulation step—which is significantly more efficient.
    # The only change from the original implementation is multiplying the batch size by `gradient_accumulation_steps`.
    # Thus, `_prepare_inputs` is called with the accumulated batch size, and it handles the splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification.As a result, some parts of the method aren't relevant to GRPO, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.gradient_accumulation_steps,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |     Accum step 0      |     Accum step 1      |
        #                                      |   GPU 0   |   GPU 1   |   GPU 0   |   GPU 1   |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     [0   0   1   1   2   2]  3   3   4   4   5   5    <- Generate for the whole accumulated batch; store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1      0   0   1   1   2   2 [ 3   3   4   4   5   5]   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     [0   0   1   1   2   2]  3   3   4   4   5   5    <- Take the stored generations and use the first slice to compute the loss
        #  num_iterations=2 ▼  1          3      0   0   1   1   2   2 [ 3   3   4   4   5   5]   <- Take the stored generations and use the second slice to compute the loss
        #
        #                      2          4     [6   6   7   7   8   8]  9   9  10  10  11  11    <- Generate for the whole accumulated batch; store the completions; use the first slice to compute the loss
        #                      2          5      6   6   7   7   8   8 [ 9   9  10  10  11  11]   <- ...
        #                                          ...
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    # def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        # """Enables gradient checkpointing for the model."""
        # # Ensure use_cache is disabled
        # model.config.use_cache = False

        # # Enable gradient checkpointing for non-PEFT models
        # model.gradient_checkpointing_enable()

        # gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        # use_reentrant = (
        #     "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        # )

        # if use_reentrant:
        #     model.enable_input_require_grads()

        # return model

    @profiling_decorator
    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep=None, **multimodal_inputs):
        # unwrap the model to access the model.model
        unwrapped_model = self.accelerator.unwrap_model(model)
        last_hidden_state = unwrapped_model(input_ids=input_ids, attention_mask=attention_mask, **multimodal_inputs, output_hidden_states=True).hidden_states[-1]
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, **multimodal_inputs) -> torch.Tensor:

        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, **multimodal_inputs
        ).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        # VLMs dosen't have a `logits_to_keep` argument, so we handle it manually.
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:]
            input_ids = input_ids[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature
        logps = selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

        return logps

    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs
    
    '''
        This function has been adapted to LVR models
    '''
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        prompts = [x["prompt"] for x in inputs]
        
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        image_inputs, video_inputs, video_kwargs = process_vision_info(prompts, return_video_kwargs=True)

        prompt_inputs = self.processing_class(
            text = prompts_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            **video_kwargs,
        )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM (deleted) 
        # Regular generation path
        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs, generation_config=self.generation_config
            )

        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # --- NEW: Build LVR mask ---
        # prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        lvr_mask = torch.ones_like(prompt_completion_ids, dtype=torch.bool)
        for b in range(prompt_completion_ids.size(0)):
            active = False
            for t in range(prompt_completion_ids.size(1)):
                tok = prompt_completion_ids[b, t].item()
                if tok == self.model.config.lvr_start_id:
                    active = True
                elif tok == self.model.config.lvr_end_id:
                    active = False
                if active:
                    lvr_mask[b, t] = False  # zero out inside LVR span


        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        final_mask = attention_mask.bool() & lvr_mask  ### << only count valid + non-LVR tokens

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in MULTIMODAL_KEYWORDS}


        def _lvr_score(model, ids, mask, logits_to_keep, **kwargs):
            if getattr(self.generation_config, "lvr_steps", 0) > 0:
                return self.score_with_lvr_replay(
                    model,
                    input_ids=ids[:, :prompt_ids.size(1)],   # only the prompt
                    attention_mask=mask[:, :prompt_ids.size(1)],
                    target_ids=ids,
                    generation_config=self.generation_config,
                    lvr_steps=[self.generation_config.lvr_steps] * ids.size(0),
                    enable_grad=False,   # for old/ref; True later in loss computation
                    **kwargs,
                )
            else:
                return self._get_per_token_logps(model, ids, mask, logits_to_keep, batch_size, **kwargs)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = _lvr_score(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, **multimodal_inputs
                )   
                old_per_token_logps = old_per_token_logps * final_mask[:, -logits_to_keep:]  ### apply mask
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                # ref_per_token_logps = self._get_per_token_logps(
                #     self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size, **multimodal_inputs
                # )
                ref_per_token_logps = _lvr_score(
                    self.ref_model, 
                    prompt_completion_ids, 
                    attention_mask, 
                    logits_to_keep, 
                    **multimodal_inputs
                )
                ref_per_token_logps = ref_per_token_logps * final_mask[:, -logits_to_keep:]
            else:
                # with self.accelerator.unwrap_model(self.model).disable_adapter():
                #     ref_per_token_logps = self._get_per_token_logps(
                #         self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size, **multimodal_inputs
                #     )
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = _lvr_score(
                        self.model, 
                        prompt_completion_ids, 
                        attention_mask, 
                        logits_to_keep, 
                        **multimodal_inputs
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "final_mask": final_mask[:, -logits_to_keep:],  ### return this for later loss computation
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "multimodal_inputs": multimodal_inputs,
        }

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            # print("computing _generate_and_score_completions!!!")
            inputs = self._generate_and_score_completions(inputs)
            # print("Adv obtained!!!")
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Unpack inputs prepared by _generate_and_score_completions
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        final_mask = inputs["final_mask"]               # NEW: excludes LVR + truncated tokens
        multimodal_inputs = inputs["multimodal_inputs"]

        old_per_token_logps = inputs["old_per_token_logps"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]

        device = prompt_ids.device
        batch_size = prompt_ids.size(0)
        logits_to_keep = completion_ids.size(1)


        # Compute the per-token log probabilities for the model
        # input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        if getattr(self.generation_config, "lvr_steps", 0) > 0:
            """This is problematic as gradients are computed multiple times"""
            # per_token_logps = self.score_with_lvr_replay(
            #     model,
            #     input_ids=prompt_ids,                  # prompt only
            #     attention_mask=prompt_mask,
            #     target_ids=prompt_completion_ids,      # full sequence (teacher forcing)
            #     generation_config=self.generation_config,
            #     lvr_steps=[self.generation_config.lvr_steps] * batch_size,
            #     enable_grad=True,                      # !!!
            #     **multimodal_inputs,
            # )

            """This method requires zero grad for lvr replay"""
            # 1) replay LVR under no_grad to reconstruct final LVR states
            lvr_states, lvr_mask, had_lvr, model_kwargs_after_lvr = self._replay_lvr_to_collect_states(
                model, 
                prompt_ids, 
                prompt_mask, 
                prompt_completion_ids, 
                multimodal_inputs,
                lvr_steps=self.generation_config.lvr_steps
            )

            # Prepare inputs for full forward (this builds the autograd graph once)
            model_kwargs = {}  # adapt: if your trainer uses special _get_initial_cache_position do that
            model_kwargs = getattr(model, "_get_initial_cache_position", lambda i, k: k)(prompt_completion_ids, model_kwargs)
            model_inputs = self.model.prepare_inputs_for_generation(prompt_completion_ids, **model_kwargs)  # fresh
            model_inputs.update(multimodal_inputs)
            model_inputs.update(
                {
                    "lvr_mask": lvr_mask, 
                    "lvr_states": lvr_states,
                    "lvr_mode_switch": None,
                    "last_position_hidden_state": None,
                    'prompt_length':prompt_ids.size(1),
                }
            )

            # Forward with grad (no torch.no_grad context)
            outputs = model(**model_inputs, return_dict=True,)

            logits = outputs.logits  # (B, T, V)
            logits = logits[:, :-1, :].contiguous()          # align to next-token
            input_tok_ids = prompt_completion_ids[:, 1:].contiguous()

            # keep only last logits_to_keep tokens (completion part)
            logits = logits[:, -logits_to_keep:, :]
            input_tok_ids = input_tok_ids[:, -logits_to_keep:]

            logits = logits / self.temperature
            per_token_logps = selective_log_softmax(logits, input_tok_ids)  # shape (B, C)


        else:
            per_token_logps = self._get_per_token_logps(
                model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                batch_size,
                **multimodal_inputs,
            )

        # ----------------------------------------------------
        # 2) Apply final_mask to exclude LVR + truncated tokens
        # Keep only completion part
        per_token_logps = per_token_logps * final_mask
        # ----------------------------------------------------
        # 3) PPO / GRPO clipped surrogate loss
        if self.num_iterations > 1:
            old_logps = old_per_token_logps * final_mask
        else:
            old_logps = per_token_logps.detach()

        # Expand advantages across tokens
        adv = advantages.to(device).unsqueeze(1)

        coef_1 = torch.exp(per_token_logps - old_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * adv
        per_token_loss2 = coef_2 * adv
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # ----------------------------------------------------
        # 4) KL penalty if beta > 0
        if self.beta != 0.0 and ref_per_token_logps is not None:
            ref_logps = ref_per_token_logps * final_mask
            per_token_kl = torch.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl
        else:
            per_token_kl = None


        # ----------------------------------------------------
        # 5) Aggregate loss depending on loss_type
        if self.loss_type == "grpo":
            loss = ((per_token_loss * final_mask).sum(-1) / final_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * final_mask).sum() / final_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * final_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Compute the KL divergence between the model and the reference model
        # if self.beta != 0.0:
        #     ref_per_token_logps = inputs["ref_per_token_logps"]
        #     per_token_kl = (
        #         torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        #     )

        # Compute the loss
        # advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        # old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        # coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        # coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        # per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        # per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        # per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        # if self.beta != 0.0:
        #     per_token_loss = per_token_loss + self.beta * per_token_kl

        # if self.loss_type == "grpo":
        #     loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        # elif self.loss_type == "bnpo":
        #     loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        # elif self.loss_type == "dr_grpo":
        #     loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        # else:
        #     raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        # if self.beta != 0.0:
        #     mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        #     self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        if per_token_kl is not None:
            mean_kl = (per_token_kl * final_mask).sum() / final_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        # low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        # high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        # clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()
        low_clip = (is_low_clipped * final_mask).sum() / final_mask.sum()
        high_clip = (is_high_clipped * final_mask).sum() / final_mask.sum()
        clip_ratio = (is_region_clipped * final_mask).sum() / final_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())


        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": self._textual_logs["prompt"],
                    "completion": self._textual_logs["completion"],
                    **self._textual_logs["rewards"],
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        # modified to support online checkpointing
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        # output_dir is the local path forcheckpoint
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # output_dir is local; now we save to cloud if needed
        if self.temp_folder:
            remote_chkpt_folder = os.path.join(self.args.remote_output_dir,checkpoint_folder)
            if remote_chkpt_folder[0] == '/':
                remote_chkpt_folder = remote_chkpt_folder[1:]       #remote pathing rules will take bucket//checkpoints, need to remove the dup
            self.oci_handler.save_checkpoint(output_dir,remote_chkpt_folder)    #save local chkpt to remote folder
            # remove the local 
            self.temp_folder.cleanup(checkpoint_name=checkpoint_folder)


        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def score_with_lvr_replay(
        self,
        model,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        target_ids: torch.LongTensor,
        generation_config,
        lvr_steps: list,
        device=None,
        # return_per_token_logps: bool = True,
        apply_temperature: float = 1.0,
        enable_grad: bool = False,   # if True, run with gradients (don't use @torch.no_grad)
        **model_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replay the LVR decoding loop and compute per-token log-probs of target_ids.
        - input_ids: (B, prompt_len) initial prompt tokens (should include any special tokens)
        - target_ids: (B, T_total) full target sequence including prompt and completion, aligned with the replay
            Usually you will pass the full input_ids + completion so that target_ids[:, step] is the "next" token at that step.
            If you prefer, target_ids can be only the *next-token* sequence aligned to model steps (i.e., shifted).
        - Returns: per_token_logps (B, total_steps) with log-prob of each token produced at each step (for tokens after the initial prompt this is the important part)
        """
        grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()

        with grad_ctx:
            if device is None:
                device = input_ids.device

            batch_size = input_ids.size(0)
            cur_input_ids = input_ids.clone().to(device)  # will be appended as we step
            # build attention mask matching cur_input_ids
            cur_attention_mask = attention_mask.clone().to(device)

            # prepare bookkeeping just like your _lvr_deocding_by_steps
            lvr_steps_orig = torch.tensor(lvr_steps, dtype=torch.int, device=device)  # can be broadcast if same
            lvr_remaining_steps = lvr_steps_orig.clone()
            lvr_mode_switch = torch.zeros(batch_size, dtype=torch.bool, device=device)
            last_position_hidden_state = None

            # where we store per-step logits for the *selected* next-token
            # max_steps = target_ids.size(1)  # safe upper bound
            per_step_logps = []
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
            this_peer_finished = False

            # To be safe, copy/paste your cache init call
            model_kwargs = {}  # adapt: if your trainer uses special _get_initial_cache_position do that
            model_kwargs = getattr(model, "_get_initial_cache_position", lambda i, k: k)(cur_input_ids, model_kwargs)

            # loop: mirror your decode loop exactly for switch/quota updates
            cur_len = cur_input_ids.shape[1]
            while not (this_peer_finished):
                # Prepare model inputs (use the same helper as in generation code)
                model_inputs = self.model.prepare_inputs_for_generation(cur_input_ids, **model_kwargs)
                # set outputs flags if necessary
                model_inputs.update({"output_attentions": generation_config.output_attentions} if getattr(generation_config, "output_attentions", False) else {})
                model_inputs.update({"output_hidden_states": generation_config.output_hidden_states} if getattr(generation_config, "output_hidden_states", False) else {})

                # pass LVR flags
                model_inputs.update({"lvr_mode_switch": lvr_mode_switch})
                model_inputs.update({"last_position_hidden_state": last_position_hidden_state})

                # call model
                outputs = model(**model_inputs, return_dict=True)

                # update model kwargs / caches like your function does
                model_kwargs = model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
                )

                # extract logits for next-token
                next_token_logits = outputs.logits[:, -1, :].to(dtype=torch.float32, device=device).clone()

                # apply same logits processor you used in generation if needed (temperature handled below)
                # compute log-probs
                logits = next_token_logits / apply_temperature
                log_probs = F.log_softmax(logits, dim=-1)  # (B, V)

                # decide the target token for this step:
                # - If target_ids is full sequence aligned with cur_input_ids, choose target_ids[:, cur_len]
                # - Otherwise adapt accordingly. We'll assume target_ids length >= cur_len+1
                if cur_len >= target_ids.size(1):
                    # nothing left to score: pad with zeros and break
                    break
                next_gold = target_ids[:, cur_len].long().to(device)  # token that *should* be produced at this step

                # gather per-example logprob for the gold token
                step_logps = log_probs.gather(dim=1, index=next_gold.unsqueeze(1)).squeeze(1)  # (B,)
                per_step_logps.append(step_logps)

                # --- update LVR counters exactly like in your code ---
                last_tokens = cur_input_ids[:, -1]
                lvr_start_switch = (last_tokens == self.model.config.lvr_start_id).to(device=device)
                new_mode_switch = lvr_mode_switch | lvr_start_switch
                just_entered = (~lvr_mode_switch) & new_mode_switch
                # still_in = lvr_mode_switch & new_mode_switch
                lvr_remaining_steps = torch.where(just_entered, lvr_steps_orig, lvr_remaining_steps)
                # lvr_remaining_steps = lvr_remaining_steps - still_in.long()
                lvr_remaining_steps = lvr_remaining_steps - lvr_mode_switch.long()
                lvr_mode_switch = new_mode_switch & (lvr_remaining_steps > 0)

                # update last_position_hidden_state from outputs (must exist in your patched forward)
                # last_position_hidden_state = getattr(outputs, "last_position_hidden_state", None)
                last_position_hidden_state = outputs.last_position_hidden_state.detach()
                # append the gold token (teacher force)
                cur_input_ids = torch.cat([cur_input_ids, next_gold.unsqueeze(1)], dim=-1)
                # extend attention mask
                cur_attention_mask = torch.cat([cur_attention_mask, torch.ones(batch_size, 1, device=device, dtype=cur_attention_mask.dtype)], dim=1)

                # update unfinished sequences: keep it consistent with your stopping rule
                # here we want to stop only after we've exhausted target tokens or all sequences hit EOS (but LVR sequences should not stop early)
                # using identical logic to your code:
                # WARNING: you may need to provide `scores` placeholder if your stopping uses it
                scores_placeholder = None
                finished_mask = torch.zeros_like(unfinished_sequences)  # we don't stop because we want full scoring; adapt if needed
                unfinished_sequences = (lvr_mode_switch | (unfinished_sequences & ~finished_mask)).long()

                this_peer_finished = (unfinished_sequences.max() == 0) or (cur_input_ids.size(1) >= target_ids.size(1))
                cur_len += 1

                # cleanup
                del outputs

            # per_step_logps is list of (B,) tensors, length Nsteps; stack into (B, Nsteps)
            if len(per_step_logps) == 0:
                per_token_logps = torch.zeros(batch_size, 0, device=device)
            else:
                per_token_logps = torch.stack(per_step_logps, dim=1)  # (B, steps)

            return per_token_logps
        
    def _compute_lvr_spans(self, prompt_completion_ids, prompt_len):
        """
        Return two long tensors (lvr_start_idx, lvr_end_idx) of shape (B,).
        If a sample has no lvr_start, start == seq_len.
        If lvr_start present and no lvr_end, end == seq_len.
        lvr tokens are considered from start index (inclusive) up to end index (exclusive).
        """
        device = prompt_completion_ids.device
        B, seq_len = prompt_completion_ids.size()
        lvr_start_idx = torch.full((B,), seq_len, dtype=torch.long, device=device)
        lvr_end_idx = torch.full((B,), seq_len, dtype=torch.long, device=device)
        start_id = self.model.config.lvr_start_id
        end_id = self.model.config.lvr_end_id

        for b in range(B):
            # scan only the completion region
            for t in range(prompt_len, seq_len):
                tok = int(prompt_completion_ids[b, t].item())
                if tok == start_id and lvr_start_idx[b] == seq_len:
                    lvr_start_idx[b] = t
                elif tok == end_id and lvr_start_idx[b] != seq_len and lvr_end_idx[b] == seq_len:
                    lvr_end_idx[b] = t
            # if we saw start but never end, end at seq_len
            if lvr_start_idx[b] != seq_len and lvr_end_idx[b] == seq_len:
                lvr_end_idx[b] = seq_len
        return lvr_start_idx, lvr_end_idx

    @torch.no_grad()
    def _replay_lvr_to_collect_states(
        self, 
        model, 
        prompt_ids, 
        prompt_mask, 
        prompt_completion_ids, 
        multimodal_inputs,
        lvr_steps
    ):
        """
        Teacher-force the *generated* tokens for the LVR windows (no gradients).
        Returns:
        - last_position_hidden_state: the final LVR state for each sample (or None for samples with no LVR)
        - had_lvr: boolean tensor (B,) True if sample used LVR at all
        This function does NOT compute any logprobs; it only advances the model to reconstruct LVR hidden states.
        """
        device = prompt_ids.device
        B, total_len = prompt_completion_ids.size()
        prompt_len = prompt_ids.size(1)
        max_completion_steps = total_len - prompt_len

        lvr_start_idx, lvr_end_idx = self._compute_lvr_spans(prompt_completion_ids, prompt_len)

        H = self.model.config.hidden_size
        lvr_states = torch.zeros(B, max_completion_steps, H, device='cpu', dtype=self.model.dtype)
        lvr_mask = torch.zeros(B, max_completion_steps, dtype=torch.bool, device='cpu')

        # initial cache / model kwargs (match generation)
        model_kwargs = {}
        model_kwargs = self.model._get_initial_cache_position(prompt_ids, model_kwargs)

        # cur_input_ids starts as prompt; we will append gold tokens teacher-forcing
        cur_input_ids = prompt_ids.clone()
        last_position_hidden_state = None

        # Calculate the end index based on configuration, not data.
        configured_end_idx = lvr_start_idx + lvr_steps

        max_inference_steps = int((lvr_end_idx - lvr_start_idx).max().item())   # this is not necessarily the actual lvr steps as model may generate lvr tokens right after the real lvr steps
        seq_len = prompt_completion_ids.size(1)

        for step in range(max_inference_steps+1):
            pos = prompt_len + step  # absolute position in prompt_completion_ids

            if pos >= seq_len:   # stop early if we reached end
                break

            # which samples are currently inside their LVR span at this absolute pos?
            in_span  = ((pos > lvr_start_idx) & (pos <= configured_end_idx)).to(device=device)

            '''
                We collect hidden states here to avoid off-by-one issues
                the inital step containing lvr_start will be skipped

            '''
            # Create a mask for samples that are in span and not the end token
            is_lvr_token_mask = (prompt_completion_ids[:, pos] != self.model.config.lvr_end_id)
            update_mask = in_span.cpu() & is_lvr_token_mask.cpu()

            # collect hidden state for *this* step if token at pos is an LVR token (not <|lvr_end|>)
            if update_mask.any():
                # Select the relevant hidden states on GPU, move to CPU, then assign
                states_to_store = last_position_hidden_state[update_mask].cpu()
                lvr_states[update_mask, step] = states_to_store
                lvr_mask[update_mask, step] = True

            # prepare model inputs for this step
            model_inputs = self.model.prepare_inputs_for_generation(cur_input_ids, **model_kwargs)
            model_inputs.update(
                {
                    "lvr_mode_switch": in_span, 
                    "last_position_hidden_state": last_position_hidden_state
                    }
            )
            model_inputs.update(multimodal_inputs)

            outputs = model(**model_inputs, return_dict=True)

            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )

            # update last_position_hidden_state (may be None for non-lvr samples)
            last_position_hidden_state = getattr(outputs, "last_position_hidden_state", last_position_hidden_state)
            
            # teacher-force append the generated gold token at pos
            next_gold = prompt_completion_ids[:, pos].unsqueeze(1)
            cur_input_ids = torch.cat([cur_input_ids, next_gold], dim=1)

            # delete outputs to free mem
            del outputs

        # had_lvr: whether start index is < seq_len
        had_lvr = (lvr_start_idx < total_len).to(device=device)
        # last_position_hidden_state is the final state after replay (or None if never produced)
        return lvr_states.to(device), lvr_mask.to(device), had_lvr, model_kwargs

