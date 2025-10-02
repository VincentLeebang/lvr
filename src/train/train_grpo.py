import sys
import os

import torch
import pathlib
from transformers import AutoProcessor, HfArgumentParser

from src.model.qwen_lvr_model import QwenWithLVR
from src.trainer import QwenGRPOTrainer
from src.dataset import make_grpo_data_module
from transformers import AutoProcessor, AutoConfig, HfArgumentParser
from src.params import DataArguments, ModelArguments, GRPOArguments
from train.train_utils import safe_save_model_for_hf_trainer
from monkey_patch_forward_lvr_rl import replace_qwen2_5_with_mixed_modality_forward_lvr_rl
from src.utils import  load_reward_funcs

from src.s3_checkpoints_lvr import OCIFolderCheckpointHandler, create_temp_dir
from src.train.monkey_patch_patch_emb import replace_qwen_2_5_vl_patch_emb
from src.train.train_utils import normalize_special_tokens

local_rank = None

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

import torch.distributed as dist
from torch.distributed import get_rank, barrier

def download_checkpoint_if_needed(remote_path, local_path, oci_handler):
    rank = get_rank() if dist.is_initialized() else 0

    if rank == 0:
        # rank 0 downloads
        oci_handler.load_checkpoint(remote_path, local_path, inference_mode=True)
        model_pth = local_path.name
    else:
        model_pth = ""

    if dist.is_initialized():
        # broadcast model path string from rank 0
        obj_list = [model_pth]
        dist.broadcast_object_list(obj_list, src=0)
        model_pth = obj_list[0]

    return model_pth



def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, GRPOArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.use_liger_loss = False


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
    
    ##if we are starting from a checkpoint
    if training_args.checkpoint_name:
        if training_args.online_checkpoint:
            # CHKPT_NAME="checkpoints_lvrHead_featureAlign/Qwen2.5-VL-7B-Instruct/BS256-LAMBDA1-LVR_HEAD_LR1e-5-MAXTOKEN{7680}/checkpoint-1578/"
            stage1_details = training_args.checkpoint_name.split('Stage1_')[-1]
            stage1_details = stage1_details.split('/')[0]

            local_pth_to_download_chkpt = create_temp_dir(base_path=os.path.join(cache_dir,model_name),prefix=f"stage1_chkpt_{stage1_details}" + '-')
            # oci_handler.load_checkpoint(training_args.checkpoint_name, local_pth_to_download_chkpt,inference_mode=True)
            model_pth = download_checkpoint_if_needed(remote_path=training_args.checkpoint_name,local_path=local_pth_to_download_chkpt,oci_handler=oci_handler)
            # model_pth = local_pth_to_download_chkpt.name
        else:
            model_pth = training_args.checkpoint_name
    else:
        model_pth = model_args.model_id
    
    # model_pth = "/dockerx/Local/users/bangzheng/Qwen2.5-VL-3B-Instruct/stage1_chkpt_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128-lxo8qmsy"
    # model_pth = "/dockerx/Local/users/bangzheng/Qwen2.5-VL-3B-Instruct/stage1_chkpt_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128-flpc_tdx"
    # d4c1
    # model_pth = "/dockerx/Local/users/bangzheng/Qwen2.5-VL-3B-Instruct/stage1_chkpt_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128-2iikdq77"
    # 9424
    # model_pth = "/dockerx/Local/users/bangzheng/Qwen2.5-VL-3B-Instruct/stage1_chkpt_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128-nmdyh7i7"
    # 7BBF
    # model_pth = "/dockerx/Local/users/bangzheng/Qwen2.5-VL-3B-Instruct/stage1_chkpt_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128-hc8f8dq3"
    # 949C
    # model_pth = "/dockerx/Local/users/bangzheng/Qwen2.5-VL-3B-Instruct/stage1_chkpt_0.5TI_mseLVRLossLambda0.1-MaxVisToken5120-MinVisToken128-_f5spje9"
    

    # double check the special_token_flag, We want to make sure lvr tokens can be output
    tokens_to_normalize = {"<|lvr_start|>", "<|lvr_end|>", "<|lvr|>", "<|lvr_latent_end|>"}
    normalize_special_tokens(model_pth,tokens_to_normalize)

    # get the model config
    config = AutoConfig.from_pretrained(model_pth,trust_remote_code=True)
    
    # Load model based on model type
    if "Qwen2.5" in model_args.model_id:
        # Patch the forward function
        replace_qwen2_5_with_mixed_modality_forward_lvr_rl()
        
        model = QwenWithLVR.from_pretrained(
            model_pth,
            config=config,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        )

        # init lvr_head
        if model_args.lvr_head:
            model._init_lvr_head(lvr_head_type =  model_args.lvr_head_type)
        
        # init latent_end_token
        if model_args.latent_end_token:
            model._init_lvr_latent_end_emb()
            model.config.loss_mode_switch_fct = training_args.loss_mode_switch_fct

        
        ''' Patch the patch-emb with fp32; Avoid edge-case nermical stability issue '''
        replace_qwen_2_5_vl_patch_emb()

    # elif "InternVL" in model_args.model_id:
    #     replace_qwen2_5_with_mixed_modality_forward_lvr(coconut=model_args.coconut,
    #                                                     lvr_head=model_args.lvr_head,
    #                                                     mode_switch_loss=training_args.mode_switch_loss,
    #                                                     latent_end_token=model_args.latent_end_token)
    #     from transformers import InternVLForConditionalGeneration
    #     model = InternVLForConditionalGeneration.from_pretrained(
    #         model_pth,
    #         torch_dtype=torch.bfloat16,
    #         low_cpu_mem_usage=True,
    #         use_flash_attn=True,
    #         trust_remote_code=True
    #     )
    #     print(1)

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
    # depends on the version of huggingface
    # model.model.visual.patch_embed.register_forward_hook(output_nan_sanitizer_hook)
    model.visual.patch_embed.register_forward_hook(output_nan_sanitizer_hook)

    # if training_args.gradient_checkpointing:
    #     model.enable_input_require_grads()
    #     training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}


    processor = AutoProcessor.from_pretrained(model_pth)


    
    dataset_module = make_grpo_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    reward_funcs = load_reward_funcs("src.train.reward_funcs")

    trainer = QwenGRPOTrainer(
        model=model,
        ref_model_pth=model_pth,
        train_dataset=dataset_module["train_dataset"],
        eval_dataset = dataset_module["eval_dataset"],
        reward_funcs=reward_funcs,
        processing_class=processor,
        args=training_args,
        temp_folder=temp_folder,
        oci_handler=oci_handler,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()