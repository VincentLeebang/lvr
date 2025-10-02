import sys
import os

import os
import torch
from transformers import AutoProcessor, AutoConfig, HfArgumentParser
from transformers import AutoTokenizer, AutoModel

from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration

from src.trainer import QwenSFTTrainer
from src.dataset import make_supervised_data_module
from src.params_vanilla import DataArguments, ModelArguments, TrainingArguments

from train.train_utils import safe_save_model_for_hf_trainer
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward

from src.s3_checkpoints_lvr import OCIFolderCheckpointHandler, create_temp_dir
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.monkey_patch_dataloader import replace_train_dataloader

local_rank = None

# For debugging only Plese comment this during training
# torch.autograd.set_detect_anomaly(True)

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    '''
        set up oci checkpointing;
        set online_checkpoint to False if you dont need
    '''
    oci_handler = None
    temp_folder = None
    if training_args.online_checkpoint:
        # oci keys
        access_key_id = os.environ.get('ACCESS_KEY_ID')
        secret_access_key = os.environ.get('SECRET_ACCESS_KEY')
        endpoint_url = os.environ.get('ENDPOINT_URL')
        bucket_name = os.environ.get('BUCKET_NAME')
        region_name = os.environ.get('REGION_NAME')

        model_name = model_args.model_id.split('/')[-1]     # "Qwen2.5-VL-7B-Instruct"
        # local cache dir and tempFile class
        cache_dir = os.getenv("CACHE_DIR")  #cache dir = "/dockerx/Local/users/bangzheng"
        # temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"
        local_model_name_or_path = create_temp_dir(base_path=os.path.join(cache_dir,model_name),prefix=training_args.run_name + '-')     
        temp_folder = local_model_name_or_path

        # remote dir
        remote_dir = training_args.output_dir  # output_dir is remote now; "/checkpoints"
        remote_dir = os.path.join(remote_dir,model_name,training_args.run_name)    # "/checkpoints/Qwen2.5-VL-7B-Instruct/run_name"
        training_args.remote_output_dir = remote_dir
        training_args.output_dir = local_model_name_or_path.name    # output_dir should always be local

        # oci handler
        oci_handler = OCIFolderCheckpointHandler(access_key_id, secret_access_key, endpoint_url, bucket_name, region_name)
    

    local_rank = training_args.local_rank

    '''
        Monkey patching model forward function with lvr
        Configure model
    '''
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # if we are starting from a checkpoint
    if training_args.checkpoint_name:
        if training_args.online_checkpoint:
            # CHKPT_NAME="checkpoints_lvrHead_featureAlign/Qwen2.5-VL-7B-Instruct/BS256-LAMBDA1-LVR_HEAD_LR1e-5-MAXTOKEN{7680}/checkpoint-1578/"
            local_pth_to_download_chkpt = create_temp_dir(base_path=os.path.join(cache_dir,model_name),prefix=f"warmed_{model_args.lvr_head_type}" + '-')
            oci_handler.load_checkpoint(training_args.checkpoint_name, local_pth_to_download_chkpt,inference_mode=True)
            
            model_pth = local_pth_to_download_chkpt.name
        else:
            model_pth = training_args.checkpoint_name
    # if its starting a new training
    else:
        model_pth = model_args.model_id
    
    # get the model config
    config = AutoConfig.from_pretrained(model_pth,trust_remote_code=True)
    
    # Load model based on model type
    if "Qwen2.5" in model_args.model_id:

        replace_qwen2_5_with_mixed_modality_forward(use_liger=False)
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )

        ''' Patch the patch-emb with fp32; Avoid edge-case nermical stability issue '''
        replace_qwen_2_5_vl_patch_emb()

    else:
        raise("Unsupported model type. At this moment, we only support Qwen2.5LM-based Qwen2.5VL series and InternVL3 series.")

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    ''' Hook the patch-emb with torch.nan_to_num() '''
    def output_nan_sanitizer_hook(module, input, output):
        if isinstance(output, torch.Tensor) and torch.isnan(output).any():
            print(f"[Sanitizer] {module.__class__.__name__}: NaN or Inf detected.")
            print(f"  Output stats - min: {output.min().item()}, max: {output.max().item()}, mean: {output.mean().item()}")
            return torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        return output
    model.visual.patch_embed.register_forward_hook(output_nan_sanitizer_hook)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # configure processors and special tokens
    processor = AutoProcessor.from_pretrained(model_args.model_id,min_pixels=data_args.image_min_pixels,max_pixels=data_args.image_max_pixels)


    '''
        Data module configurations
        use data packing for faster training due to the random input lengths of LVR
    '''
    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args,)
    
    # tempFolder = temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"
    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        temp_folder=temp_folder,
        oci_handler=oci_handler,
        **data_module
    )

    trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
