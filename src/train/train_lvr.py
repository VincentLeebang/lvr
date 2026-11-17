import os
import torch

from transformers import AutoConfig, AutoProcessor, HfArgumentParser

from src.model.qwen_lvr_model import QwenWithLVR
from src.trainer import QwenLVRSFTTrainer
from src.dataset import (
    make_supervised_data_module_lvr,
    make_packed_supervised_data_module_lvr,
)
from src.params import DataArguments, ModelArguments, TrainingArguments

from train.train_utils import safe_save_model_for_hf_trainer
from src.train.monkey_patch_forward_lvr import (
    replace_qwen2_5_with_mixed_modality_forward_lvr,
)
from src.s3_checkpoints_lvr import OCIFolderCheckpointHandler, create_temp_dir
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.monkey_patch_dataloader import replace_train_dataloader


local_rank = None


def rank0_print(*args):
    if local_rank in (0, "0", None, -1):
        print(*args)


def set_requires_grad(parameters, requires_grad):
    for parameter in parameters:
        parameter.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    """
    Qwen2.5-VL stores its vision tower at model.model.visual.
    """
    vision_tower = model.model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    set_requires_grad(
        vision_tower.parameters(),
        not training_args.freeze_vision_tower,
    )

    # The merger can be frozen/unfrozen independently of the vision tower.
    set_requires_grad(
        vision_tower.merger.parameters(),
        not training_args.freeze_merger,
    )


def configure_llm(model, training_args):
    """
    Freeze/unfreeze the language model and LM head.
    Vision parameters are configured separately afterwards.
    """
    set_requires_grad(
        model.lm_head.parameters(),
        not training_args.freeze_llm,
    )

    set_requires_grad(
        model.model.parameters(),
        not training_args.freeze_llm,
    )


def setup_oci_checkpointing(model_args, training_args):
    """
    Configure optional OCI checkpoint handling.
    Returns (handler, temporary_folder, cache_dir, model_name).
    """
    if not training_args.online_checkpoint:
        return None, None, None, None

    access_key_id = os.environ.get("ACCESS_KEY_ID")
    secret_access_key = os.environ.get("SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("ENDPOINT_URL")
    bucket_name = os.environ.get("BUCKET_NAME")
    region_name = os.environ.get("REGION_NAME")
    cache_dir = os.environ.get("CACHE_DIR")

    required_values = {
        "ACCESS_KEY_ID": access_key_id,
        "SECRET_ACCESS_KEY": secret_access_key,
        "ENDPOINT_URL": endpoint_url,
        "BUCKET_NAME": bucket_name,
        "REGION_NAME": region_name,
        "CACHE_DIR": cache_dir,
    }
    missing = [key for key, value in required_values.items() if not value]
    if missing:
        raise ValueError(
            "Online checkpointing is enabled, but these environment "
            f"variables are missing: {', '.join(missing)}"
        )

    model_name = model_args.model_id.split("/")[-1]
    local_model_folder = create_temp_dir(
        base_path=os.path.join(cache_dir, model_name),
        prefix=training_args.run_name + "-",
    )

    remote_output_dir = os.path.join(
        training_args.output_dir,
        model_name,
        training_args.run_name,
    )
    training_args.remote_output_dir = remote_output_dir
    training_args.output_dir = str(local_model_folder.name)

    handler = OCIFolderCheckpointHandler(
        access_key_id,
        secret_access_key,
        endpoint_url,
        bucket_name,
        region_name,
    )

    return handler, local_model_folder, cache_dir, model_name


def load_lvr_model(model_args, data_args, training_args, oci_handler,
                   cache_dir, model_name):
    """
    Load the Qwen2.5-VL model and apply the LVR forward patch.
    """
    if "Qwen2.5" not in model_args.model_id:
        raise ValueError(
            "Unsupported model. This training script currently supports "
            "Qwen2.5-VL checkpoints."
        )

    if training_args.checkpoint_name:
        if training_args.online_checkpoint:
            checkpoint_folder = create_temp_dir(
                base_path=os.path.join(cache_dir, model_name),
                prefix=f"warmed_{model_args.lvr_head_type}-",
            )
            oci_handler.load_checkpoint(
                training_args.checkpoint_name,
                checkpoint_folder,
                inference_mode=True,
            )
            model_path = str(checkpoint_folder.name)
        else:
            model_path = training_args.checkpoint_name
    else:
        model_path = model_args.model_id

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    config.latent_end_token = model_args.latent_end_token
    config.lvr_head = model_args.lvr_head
    config.lvr_head_type = model_args.lvr_head_type

    # Apply the custom Qwen2.5-VL forward implementation before loading weights.
    replace_qwen2_5_with_mixed_modality_forward_lvr(
        coconut=model_args.coconut,
        lvr_head=model_args.lvr_head,
        mode_switch_loss=training_args.mode_switch_loss,
        latent_end_token=model_args.latent_end_token,
    )

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else torch.bfloat16
        if training_args.bf16
        else torch.float32
    )

    model = QwenWithLVR.from_pretrained(
        model_path,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation=(
            "sdpa"
            if training_args.disable_flash_attn2
            else "flash_attention_2"
        ),
    )

    if model_args.latent_end_token:
        model.config.loss_mode_switch_fct = training_args.loss_mode_switch_fct

    # Patch Qwen2.5-VL patch embedding for fp32 numerical stability.
    replace_qwen_2_5_vl_patch_emb()

    return model, compute_dtype


def configure_model_training(model, training_args, compute_dtype):
    model.config.use_cache = False

    configure_llm(model, training_args)

    configure_vision_tower(
        model,
        training_args,
        compute_dtype,
        training_args.device,
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": True
        }


def configure_lvr_tokens(model, processor, training_args):
    """
    Add LVR special tokens to the processor tokenizer and synchronize
    the model embedding table with the tokenizer vocabulary.
    """
    lvr_special_tokens = [
        "<|lvr_start|>",
        "<|lvr|>",
        "<|lvr_latent_end|>",
        "<|lvr_end|>",
    ]
    processor.tokenizer.add_tokens(
        lvr_special_tokens,
        special_tokens=True,
    )

    model.config.lvr_start_id = processor.tokenizer.convert_tokens_to_ids(
        "<|lvr_start|>"
    )
    model.config.lvr_id = processor.tokenizer.convert_tokens_to_ids(
        "<|lvr|>"
    )
    model.config.lvr_latent_end_id = processor.tokenizer.convert_tokens_to_ids(
        "<|lvr_latent_end|>"
    )
    model.config.lvr_end_id = processor.tokenizer.convert_tokens_to_ids(
        "<|lvr_end|>"
    )

    current_embedding_size = model.get_input_embeddings().num_embeddings
    tokenizer_size = len(processor.tokenizer)

    if tokenizer_size > current_embedding_size:
        model.resize_token_embeddings(tokenizer_size)

    model.config.loss_lvr_fct = training_args.loss_lvr_fct


def create_data_module(model_args, data_args, training_args, processor):
    """
    Create either the regular LVR dataset or the supported packed dataset.
    The fixed-token packed dataset is not present in this repository.
    """
    if training_args.enable_data_packing:
        training_args.per_device_train_batch_size = 1

        if model_args.max_lvr_tokens is not None:
            raise NotImplementedError(
                "max_lvr_tokens was requested, but this repository does not "
                "provide make_packed_supervised_data_module_lvr_fixedToken. "
                "Set max_lvr_tokens to None or disable data packing."
            )

        data_module, total_data_len = make_packed_supervised_data_module_lvr(
            model_id=model_args.model_id,
            processor=processor,
            data_args=data_args,
            training_args=training_args,
            latent_end_token=model_args.latent_end_token,
        )

        if not training_args.max_steps:
            denominator = (
                training_args.gradient_accumulation_steps
                * training_args.world_size
                * training_args.per_device_train_batch_size
            )
            if denominator <= 0:
                raise ValueError(
                    "Cannot calculate max_steps because the effective "
                    "training batch-size denominator is not positive."
                )
            training_args.max_steps = max(1, total_data_len // denominator)

        # Required by this repository's packed-data loading implementation.
        replace_train_dataloader()
        return data_module

    # The regular dataset loader accepts exactly three arguments.
    return make_supervised_data_module_lvr(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args,
    )


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    local_rank = training_args.local_rank

    if "Qwen2.5" not in model_args.model_id:
        raise ValueError(
            "Please use Qwen/Qwen2.5-VL-7B-Instruct or a compatible "
            "Qwen2.5-VL checkpoint."
        )

    # Set up optional OCI checkpointing.
    oci_handler, temp_folder, cache_dir, model_name = (
        setup_oci_checkpointing(model_args, training_args)
    )

    # Load model and apply LVR forward patch.
    model, compute_dtype = load_lvr_model(
        model_args,
        data_args,
        training_args,
        oci_handler,
        cache_dir,
        model_name,
    )

    # Configure trainable parameters and gradient checkpointing.
    configure_model_training(
        model,
        training_args,
        compute_dtype,
    )

    # Load processor with the same model family as the model checkpoint.
    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        min_pixels=data_args.image_min_pixels,
        max_pixels=data_args.image_max_pixels,
    )

    # Register LVR tokens and resize embeddings if required.
    configure_lvr_tokens(
        model,
        processor,
        training_args,
    )

    # Build the selected dataset module.
    data_module = create_data_module(
        model_args,
        data_args,
        training_args,
        processor,
    )

    # Preserve the existing trainer and checkpoint-saving workflow.
    trainer = QwenLVRSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        temp_folder=temp_folder,
        oci_handler=oci_handler,
        **data_module,
    )

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(
        trainer,
        output_dir=training_args.output_dir,
    )


if __name__ == "__main__":
    train()
