import transformers
import torch
import logging


def maybe_zero_3(param, ignore_status=False, name=None, device=torch.device('cpu')):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if type(device) is str:
        device = torch.device(device)
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach()
    else:
        param = param.detach()
    if device == param.device:
        return param.clone()
    else:
        return param.to(device)

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)

import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import Set, Union

def normalize_special_tokens(
    tokenizer_path: str, 
    tokens_to_normalize: Set[str]
) -> Union[PreTrainedTokenizerFast, None]:
    """
    Modifies tokenizer configuration files to change specified special tokens into normal tokens.

    This function directly edits 'tokenizer.json' and 'tokenizer_config.json' to ensure
    the change is permanent and correctly loaded.

    Args:
        tokenizer_path: The file path to the directory containing the tokenizer files.
        tokens_to_normalize: A set of token strings (e.g., {"<|user|>", "<|end|>"}) 
                             that should no longer be treated as special.

    Returns:
        The reloaded tokenizer object with the changes applied, or None if the path is invalid.
    """
    if not os.path.isdir(tokenizer_path):
        print(f"Error: Directory not found at '{tokenizer_path}'")
        return None

    print(f"Attempting to normalize: {tokens_to_normalize} in '{tokenizer_path}'")

    # --- Step 1: Edit tokenizer.json (for fast tokenizers) ---
    tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json")
    try:
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)

        changed_count = 0
        if 'added_tokens' in tokenizer_data:
            for idx,token_info in enumerate(tokenizer_data['added_tokens']):
                if token_info.get("content") in tokens_to_normalize and token_info.get("special") is True:
                    tokenizer_data['added_tokens'][idx]["special"] = False
                    changed_count += 1
                    print(f"- Found and changed '{token_info['content']}' in tokenizer.json")
        
        if changed_count > 0:
            with open(tokenizer_json_path, "w", encoding="utf-8") as f:
                json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved modified tokenizer.json")
        else:
            print("No matching special tokens were found to change in tokenizer.json.")
    except FileNotFoundError:
        print(f"Warning: 'tokenizer.json' not found. This may be a legacy (slow) tokenizer.")
    except Exception as e:
        print(f"An error occurred while processing tokenizer.json: {e}")


    # --- Step 2: Edit tokenizer_config.json ---
    config_json_path = os.path.join(tokenizer_path, "tokenizer_config.json")
    try:
        with open(config_json_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        for tok_id in config_data['added_tokens_decoder']:
            if config_data['added_tokens_decoder'][tok_id]['content'] in tokens_to_normalize:
                config_data['added_tokens_decoder'][tok_id]['special'] = False
        with open(config_json_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print("✅ Successfully cleaned and saved tokenizer_config.json")
    except FileNotFoundError:
        print(f"Warning: 'tokenizer.json' not found. This may be a legacy (slow) tokenizer.")
    except Exception as e:
        print(f"An error occurred while processing tokenizer.json: {e}")