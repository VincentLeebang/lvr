# Latent Visual Reasoning

This repository contains a script for training [Latent Visual Reasoning](https://www.arxiv.org/abs/2509.24251) based on [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

## Update

- [2025/10/02] 🔥Code base released.

## Table of Contents

- [Latent Visual Reasoning](#latent-visual-reasoning)
  - [Update](#update)
  - [Table of Contents](#table-of-contents)
  - [Supported Features](#supported-features)
  - [Environments](#environments)
  - [Model Weights](#model-weights)
  - [Dataset Preparation](#dataset-preparation)
  - [Training LVR](#training-lvr)
    - [Stage-1 SFT](#full-finetuning)
    - [Stage-2 GRPO<sub>latent</sub>](#stage-2-GRPO<sub>latent</sub>)
  - [Inference](#inference)
  - [TODO](#todo)
  - [Known Issues](#known-issues)
  - [License](#license)
  - [Citation](#citation)
  - [Acknowledgement](#acknowledgement)

## Supported Features

- Deepspeed
- Full-finetuning
- GRPO<sub>latent</sub>


### Environments

```bash
conda env create -f environment.yaml
conda activate train
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation
```

**Note:** You should install flash-attn after installing the other packages.<br>
**Note:** This project is forked from [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune) where you can find more instructions on environments.

## Model Weights
Model checkpoints are accessible from [vincentleebang/LVR-7B](https://huggingface.co/vincentleebang/LVR-7B)

## Dataset Preparation

### Data ###
Please download the training data through [this link](https://drive.google.com/file/d/1RUbKMQU3H7u8iWDqpDv8aiYDMlUaDIlE/view?usp=sharing) where we provide formatted training data for Latent Visual Reasoning.

To train LVR with your own data: The script requires a dataset formatted according to the LLaVA specification. The dataset should be a JSON file where each entry contains information about conversations and images. Ensure that the image paths in the dataset match the provided `--image_folder`.<br>

**Please see the example below and follow format your data. The \<image\> and \<lvr\> token are placeholders for data collation.**<br>

<details>
<summary>Example for Stage-1 SFT dataset</summary>

```json
[
  {
    "dataset": "flickr30k", 
    "split": "train", 
    "question_id": 31593, 
    "image": ["viscot/flickr30k/2618322793.jpg"], 
    "conversations": [
          {
            "from": "human", 
            "value": "<image>\nCan you describe the lower apparel of the child on the swing?\nProvide a short and direct response."
          }, 
          {
            "from": "gpt", "value": "<lvr>\n<answer> The child on the swing is wearing dark blue denim shorts. </answer>"
          }
    ], 
    "bboxes": [[0.382, 0.456, 0.718, 0.656]]
  }
  ...
]
```

</details>


<details>
<summary>Example for Stage-2 GRPO<sub>latent</sub> dataset</summary>

```json
[
  {
    "dataset": "ViRL39K", 
    "id": "MMK12-abc85ebc-7a73-4d55-80a8-ca256f84069c", 
    "image": "ViRL39K/MMK12-abc85ebc-7a73-4d55-80a8-ca256f84069c-0.png", 
    "conversations": [
      {
        "from": "human", 
        "value": "As shown in the figure, $$AB \\perp CD$$ at point $$C$$, $$CE \\perp CF$$, then there are ___ pairs of complementary angles in the figure."
      }, 
      {
        "from": "gpt", 
        "value": "<answer>4</answer>"
      }
    ]
  }
  ...
]
```

**Note:** You should remove all `<image>`tokens in your dataset. It works a bit different with other training methods.

</details>

<br><br>

## Training LVR

**Note:** We use a data packing strategy adapted from InternVL, where short instances are packed together while long instances are left unaltered to maximize GPU utilization. You can enable this feature by setting `--enable_data_packing True`.<br><br>
**Tip:** The 3D convolution module in Qwen2.5-VL's visual encoding process can introduce NaN due to numeric stability. Please refer to src/train/monkey_patch_patch_emb.py.

To run the training script, use the following command:

### Stage-1 SFT

```bash
bash scripts/finetune_lvr_stage1_7b.sh
```

<details>
<summary>Training arguments</summary>

- `--deepspeed` (str): Path to DeepSpeed config file (default: "scripts/zero2.json").
- `--data_path` (str): Path to the LLaVA formatted training data (a JSON file). **(Required)**
- `--image_folder` (str): Path to the images folder as referenced in the LLaVA formatted training data. **(Required)**
- `--model_id` (str): Path to the Qwen2-VL model. **(Required)**
- `--output_dir` (str): Output directory for model checkpoints
- `--num_train_epochs` (int): Number of training epochs (default: 1).
- `--per_device_train_batch_size` (int): Training batch size per GPU per forwarding step.
- `--gradient_accumulation_steps` (int): Gradient accumulation steps (default: 4).
- `--freeze_vision_tower` (bool): Option to freeze vision_model (default: False).
- `--freeze_llm` (bool): Option to freeze LLM (default: False).
- `--freeze_merger` (bool): Option to tune projector (default: False).
- `--vision_lr` (float): Learning rate for vision_model.
- `--merger_lr` (float): Learning rate for merger(projector).
- `--learning_rate` (float): Learning rate for language module.
- `--bf16` (bool): Option for using bfloat16.
- `--fp16` (bool): Option for using fp16.
- `--image_min_pixels` (int): Option for minimum input tokens for image.
- `--image_max_pixles` (int): Option for maximum maxmimum tokens for image.
- `--max_seq_length` (int): Maximum sequence length (default: 32K).
- `--bits` (int): Quantization bits (default: 16).
- `--disable_flash_attn2` (bool): Disable Flash Attention 2.
- `--report_to` (str): Reporting tool (choices: 'tensorboard', 'wandb', 'none') (default: 'tensorboard').
- `--logging_dir` (str): Logging directory (default: "./tf-logs").
- `--logging_steps` (int): Logging steps (default: 1).
- `--dataloader_num_workers` (int): Number of data loader workers (default: 4).
- `--precompute_ref_log_probs` (bool): Wheter to precompute the reference log probs (default: False)
- `--beta` (float): The beta value for DPO (default: 0.1)

</details>

### Stage-2 GRPO<sub>latent</sub>

```bash
bash scripts/finetune_lvr_stage2_7b.sh
```


<br>

### Prerequisites

| What                      | Where                       | Notes                                                                                       |
| ------------------------- | --------------------------- | ------------------------------------------------------------------------------------------- |
| **Reward functions**      | `src/train/reward_funcs.py` | Add any function that ends with `_reward`. The training script picks them up automatically. |
| **Custom system prompts** | `src/constants.py`          | Append your own prompt strings here.                                                        |

You could start training using this script.<br>
Before training, **Please check the dataset format once more.** The format is a bit different from other training methods.
Most of the training arugments are same as SFT, but few other arguments are added for GRPO training.

<details>
<summary>Training arguments</summary>

- `--temperature` (float): Generation config (default: 0.9)  <- LVR is quite sensitive to temperature during RL. Too large or too small temperature may 
- `--top_p` (float): Generation config (default: 1.0)
- `--top_k` (int): Generation config (default: 50)
- `--min_p` (float): Generation config (default: None)
- `--repetition_penalty` (float): Generation config (default: 1.0)
- `--max_completion_length` (int): Max length for the completion (default: 256)
- `--max_prompt_length` (int): Max length for the prompt (default: 512)
- `--beta` (float): KL Coefficient. (default: 0.04)

</details>

**Note:** **Liger GRPO loss** and **vLLM back-end** are not yet supported. Both will be added soon.

## Inference

We provide a evaluation file in evaluation/ which by default uses max-step decoding. All variants of decoding strategies are in src/model/qwen_lvr_model.py for reference.


## TODO

- [ ] Upload docker file for easy deployment

## Known Issues

- Transformer version mismatch: the RL bode base may require transformers>=4.54.0 where they updated the abstract model architechture.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this repository useful in your project, please consider giving a :star: and citing:

```bibtex
@misc{li2025lvr,
      title={Latent Visual Reasoning}, 
      author={Bangzheng Li and Ximeng Sun and Jiang Liu and Ze Wang and Jialian Wu and Xiaodong Yu and Hao Chen and Emad Barsoum and Muhao Chen and Zicheng Liu},
      year={2025},
      journal={arXiv preprint arXiv:2509.24251}
}
```

## Acknowledgement

This project is based on

- [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune): An open-source project for finetuning Qwen-2-VL/Qwen-2.5VL models.
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen3-VL): MLLM series from Qwen family. 
- [InternVL](https://github.com/OpenGVLab/InternVL/tree/main): Open-source MLLM family by Shanghai AI Lab. They also opensourced awesome tools for training MLLMs.
