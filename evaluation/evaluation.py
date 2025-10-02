import sys
import os
import torch

from datasets import load_dataset
import json
from tqdm import tqdm
from src.model.qwen_lvr_model import QwenWithLVR
from src.s3_checkpoints_lvr import OCIFolderCheckpointHandler, create_temp_dir
from transformers import AutoTokenizer, AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
from safetensors import safe_open

from qwen_vl_utils import process_vision_info
from PIL import Image

CACHE_DIR = "/dockerx/Local/users/bangzheng"
access_key_id = "cd2fce68e7166482654fe48ad7a49a2edf12b7ec"
secret_access_key = "bLInsfvXCRP1T8GhAL89BrZ24b6w+aKD3rjxkXsSgIQ="
endpoint_url = "https://idhpuomb10ix.compat.objectstorage.us-ashburn-1.oraclecloud.com"
bucket_name = "bangzhengli"
region_name = "us-ashburn-1"

oci_handler = OCIFolderCheckpointHandler(access_key_id, secret_access_key, endpoint_url, bucket_name, region_name)

from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr

STEP_LIST = [4,8,16]
# ==== Config ====


CHKPT_PATHS = ""

DATASET_CONFIG = {
    'blink': {
        "loader": lambda gen_w_head,run_name,decoding_strategy: load_blink_dataset(gen_w_head,run_name,decoding_strategy),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy: evaluate_blink(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy),
    },
    "vstar": {
        "loader": lambda gen_w_head,run_name,decoding_strategy: load_vstar_dataset(gen_w_head,run_name,decoding_strategy),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy: evaluate_vstar(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy),
    },
    "MMVP": {
        "loader": lambda gen_w_head,run_name,decoding_strategy: load_mmvp_dataset(gen_w_head,run_name,decoding_strategy),
        "evaluator": lambda model,proc,data,img_dir,out_dir,ds_name,decoding_strategy: evaluate_mmvp(model, proc, data, img_dir, out_dir, ds_name, decoding_strategy),
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Core utilities ====

def accuracy_reward(response: str, ground_truth: str) -> float:
    # content_match = re.search(r"<answer>(.*?)</answer>", response)
    # given_answer = content_match.group(1).strip() if content_match else response.strip()
    given_answer = response.split('<answer>')[-1]
    given_answer = given_answer.split('</answer')[0].strip()
    if " " in given_answer:
        given_answer = given_answer.split(" ")[0]
    if len(given_answer) >1:
        given_answer = given_answer[0]
    return given_answer == ground_truth

def get_task_instruction(bench_name):
    if bench_name == "vstar":
        return "\nAnswer with the option's letter from the given choices directly."
    elif bench_name == "mmvp":
        return "\nAnswer with the option's letter from the given choices directly."
    elif bench_name == "blink":
        return "\nAnswer with the option's letter from the given choices directly."
    else:
        raise ValueError(f"Unknown benchmark: {bench_name}")

def create_messages(img_path,question):

    if not isinstance(img_path, list):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
    else:
        vision_content = []
        for ip in img_path:
            vision_content.append({
                        "type": "image",
                        "image": ip,
                    })
        vision_content.append({"type": "text", "text": question})
        messages = [
            {
                "role": "user",
                "content": vision_content,
            }
        ]
    return messages

def load_model_and_processor(chkpt_pth):
    details_list = chkpt_pth.split('/')[0:]
    run_name = '_'.join(details_list)

    config = AutoConfig.from_pretrained(chkpt_pth)

    replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=True,lvr_head=config.lvr_head)

    model = QwenWithLVR.from_pretrained(
        chkpt_pth,
        config=config,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto",)

    processor = AutoProcessor.from_pretrained(chkpt_pth)

    return model, processor, run_name

def run_inference(model, processor,img_path,text,steps,decoding_strategy):
    messages = create_messages(img_path,text)
    text_formatted = processor.apply_chat_template( messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_formatted],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    lvr_steps = [steps]
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512,decoding_strategy=decoding_strategy,lvr_steps=lvr_steps)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    return output_text

# ==== Evaluation ====

def evaluate_vstar(
        model, 
        processor, 
        dataset,
        image_dir,
        out_dir,
        ds_name,
        decoding_strategy="steps",
        ):
    print(f"Evaluating VSTAR with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir,exist_ok=True)
    task_instruction = get_task_instruction("vstar")
    step2results_category = {}
    step2results_overall = {}
    for steps in STEP_LIST:
        step2results_category[steps] = {}
        if len(task_instruction)>0 and task_instruction[0] != ' ':
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}.json")
        else:
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}_noTaskInstruction.json")
        total, correct = 0, 0
        res_by_category = {}
        if os.path.exists(out_file):
            # run_details = "-".join(out_file.split('/')[-3:])
            # print(f"result file existed, loading results for evaluation for {run_details}:")
            with open(out_file, "r") as f:
                result = json.load(f)
            # Recompute accuracy
            for res in result:
                if "category" not in res:
                    res["category"] = "direct_attributes" if int(res["id"]) <= 114 else "relative_position"
                if res["category"] not in res_by_category:
                    res_by_category[res["category"]] = {"total":0,"correct":0}
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[res["category"]]["correct"] += 1
                total += 1
                res_by_category[res["category"]]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            result = []
            for dat in tqdm(dataset,desc=f"Evaluating Vstar, decoding by steps={steps}"):
                img_path = dat['image']
                img_path = os.path.join(image_dir,img_path)
                text = dat['text'] + task_instruction
                outputs = run_inference(model, processor,img_path,text,steps,decoding_strategy)

                res = {
                    'id': dat['question_id'],
                    'prediction': outputs,
                    'label': dat['label'],
                    'category': dat['category']
                }
                result.append(res)
                if accuracy_reward(outputs[0], dat['label']):
                    correct += 1
                total += 1
            json.dump(result,open(out_file,'w+'),indent=2)
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")
    # print overall
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))
    # print by category
    for category in ["direct_attributes","relative_position"]:
        print(f"Category: {category}")
        res  = []
        for steps in step2results_category:
            res_by_category = step2results_category[steps]
            if category in res_by_category:
                cat_total = res_by_category[category]["total"]
                cat_correct = res_by_category[category]["correct"]
                res.append(cat_correct/cat_total)
        print(",".join([f"{items*100:.2f}" for items in res]))
    return

def evaluate_mmvp(
        model, 
        processor, 
        dataset,
        image_dir,
        out_dir,
        ds_name,
        decoding_strategy="steps",
        ):
    print(f"Evaluating MMVP with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir,exist_ok=True)
    task_instruction = get_task_instruction("mmvp")

    for steps in STEP_LIST:
        if len(task_instruction)>0 and task_instruction[0] != ' ':
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}.json")
        else:
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}_noTaskInstruction.json")
        total, correct = 0, 0
        if os.path.exists(out_file):
            run_details = "-".join(out_file.split('/')[-3:])
            # print(f"result file existed, loading results for evaluation for {run_details}:")
            with open(out_file, "r") as f:
                result = json.load(f)
            # Recompute accuracy
            for res in result:
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                total += 1
        else:
            result = []
            for dat in tqdm(dataset,desc=f"Evaluating MMVP, decoding by steps={steps}"):
                img_path = dat['image']
                img_path = os.path.join(image_dir,img_path)
                text = dat['query'].replace('(a)','A.')
                text = text.replace('(b)','B.')
                text = text + task_instruction
                outputs = run_inference(model, processor,img_path,text,steps,decoding_strategy)
                if dat['label'] in ['(a)','(b)']:
                    dat['label'] = dat['label'].strip().upper()[1]
                res = {
                    'id': dat['question_id'],
                    'prediction': outputs,
                    'label': dat['label']
                }
                result.append(res)
                if accuracy_reward(outputs[0], dat['label']):
                    correct += 1
                total += 1
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")
        json.dump(result,open(out_file,'w+'),indent=2)
    return {"dummy_metric": 0}

def evaluate_blink(
        model, 
        processor, 
        dataset,
        image_dir,
        out_dir,
        ds_name,
        decoding_strategy="steps",
        ):
    print(f"Evaluating BLINK with decoding strategy: {decoding_strategy}")
    os.makedirs(out_dir,exist_ok=True)
    task_instruction = get_task_instruction("blink")
    step2results_category = {}
    step2results_overall = {}
    for steps in STEP_LIST:
        step2results_category[steps] = {}
        if len(task_instruction)>0 and task_instruction[0] != ' ':
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}.json")
        else:
            out_file = os.path.join(out_dir,f"{decoding_strategy}{steps:03d}_noTaskInstruction.json")
        total, correct = 0, 0
        res_by_category = {}
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                result = json.load(f)
            # Recompute accuracy
            for res in result:
                if res["category"] not in res_by_category:
                    res_by_category[res["category"]] = {"total":0,"correct":0}
                if accuracy_reward(res["prediction"][0], res["label"]):
                    correct += 1
                    res_by_category[res["category"]]["correct"] += 1
                total += 1
                res_by_category[res["category"]]["total"] += 1
            step2results_category[steps] = res_by_category
            step2results_overall[steps] = {"total": total, "correct": correct}
        else:
            result = []
            for dat in tqdm(dataset,desc=f"Evaluating BLINK, decoding by steps={steps}"):
                img_path = dat['image']
                text = dat['query'] + task_instruction
                outputs = run_inference(model, processor,img_path,text,steps,decoding_strategy)

                res = {
                    'id': dat['question_id'],
                    'prediction': outputs,
                    'label': dat['label'],
                    'category': dat['category']
                }
                result.append(res)
                if accuracy_reward(outputs[0], dat['label']):
                    correct += 1
                total += 1
            json.dump(result,open(out_file,'w+'),indent=2)
        print(f"Steps: {steps} - Accuracy: {correct}/{total} = {correct/total*100:.2f}")
    # print overall
    print("Overall accuracy by steps:")
    print(",".join([f"{items['correct']/items['total']*100:.2f}" for items in step2results_overall.values()]))
    # print by category
    for category in [ 'Counting','IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization', 'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence', 'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity']:
        # print(f"Category: {category}")
        res  = []
        for steps in step2results_category:
            res_by_category = step2results_category[steps]
            if category in res_by_category:
                cat_total = res_by_category[category]["total"]
                cat_correct = res_by_category[category]["correct"]
                res.append(cat_correct/cat_total)
        print(category+','+",".join([f"{items*100:.2f}" for items in res]))
    return

"""Data Loaders"""
# ==== Example dataset loader stubs ====
def load_vstar_dataset(gen_w_head,run_name,decoding_strategy):
    ds_name = "vstar"
    # Implement loading from files or a proper library
    ds = load_dataset("craigwu/vstar_bench")
    image_dir = "/dockerx/bangzhli/projects/LVR-Finetune/evaluation/vstar_bench"

    if gen_w_head:
        out_dir = os.path.join("/dockerx/bangzhli/projects/LVR-Finetune/evaluation/vstar_bench/results",f"decoding_by_{decoding_strategy}","GenWHead"+run_name)

    else:
        out_dir = os.path.join("/dockerx/bangzhli/projects/LVR-Finetune/evaluation/vstar_bench/results",f"decoding_by_{decoding_strategy}",run_name)

    return ds['test'],image_dir,out_dir,ds_name

import csv
def load_mmvp_dataset(gen_w_head,run_name,decoding_strategy):
    ds_name = "mmvp"
    csv_file = "/dockerx/bangzhli/projects/LVR-Finetune/evaluation/MMVP/Questions.csv"
    image_dir = "/dockerx/bangzhli/projects/LVR-Finetune/evaluation/MMVP/MMVP_Images"

    data = []

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean up the row if needed (split options, etc.)
            idx = int(row["Index"])
            item = {
                "question_id": idx,
                'image': f"{idx}.jpg",
                "query": row["Question"] + '\nOptions:\n' + row["Options"], 
                "label": row["Correct Answer"]
            }
            data.append(item)

    if gen_w_head:
        out_dir = os.path.join("/dockerx/bangzhli/projects/LVR-Finetune/evaluation/MMVP/results",f"decoding_by_{decoding_strategy}","GenWHead"+run_name)

    else:
        out_dir = os.path.join("/dockerx/bangzhli/projects/LVR-Finetune/evaluation/MMVP/results",f"decoding_by_{decoding_strategy}",run_name)

    return data,image_dir,out_dir,ds_name

from datasets import get_dataset_config_names
import string
def load_blink_dataset(gen_w_head,run_name,decoding_strategy):
    ds_name = "blink"
    configs = [ 'Counting','IQ_Test', 'Jigsaw', 'Relative_Reflectance', 'Spatial_Relation']
    all_datasets = {}
    for config in configs:
        all_datasets[config] = load_dataset("BLINK-Benchmark/BLINK", config)
    # ds = load_dataset("BLINK-Benchmark/BLINK")
    image_dir = None


    processed_data = []
    for config in all_datasets:
        ds = all_datasets[config]['val']
        for dat in ds:
            idx = dat["idx"]
            choices = dat["choices"]
            letters = string.ascii_uppercase 
            paired = list(zip(letters, choices))
            option_string = ""
            for letter, choice in paired:
                option_string += f"{letter}. {choice}\n"
            if len(dat['answer']) >1:
                ans = dat['answer'][1].upper()
            else:
                ans = dat['answer'][0].upper()
            images = []
            for k in ['image_1','image_2','image_3','image_4']:
                if k in dat and dat[k] is not None:
                    images.append(dat[k])
            question = dat['question'] + "\nOptions:\n" + option_string
            buffer = {
                "question_id": idx,
                "image": images,
                "query": question,
                "label": ans,
                "category": config
            }
            processed_data.append(buffer)

    if gen_w_head:
        out_dir = os.path.join("/dockerx/bangzhli/projects/LVR-Finetune/evaluation/blink/results",f"decoding_by_{decoding_strategy}","GenWHead"+run_name)

    else:
        out_dir = os.path.join("/dockerx/bangzhli/projects/LVR-Finetune/evaluation/blink/results",f"decoding_by_{decoding_strategy}",run_name)

    return processed_data,image_dir,out_dir,ds_name
# ==== Main evaluation ====

def main():
    DECODING_STRATEGY = "steps"  # or "latent"
    for checkpoint_dir in CHKPT_PATHS:
        model, processor, run_name = load_model_and_processor(checkpoint_dir)
        gen_w_head = model.config.lvr_head

        print("\n" + "="*128 + "\nEvaluating model:\n" +  f"{checkpoint_dir} \n"+"="*128 + "\n")
        for bench_name, cfg in DATASET_CONFIG.items():
            print("<"*64 + f" {bench_name} evaluation " + ">"*64)
            dataset, image_dir, out_dir, ds_name= cfg["loader"](gen_w_head,run_name,decoding_strategy=DECODING_STRATEGY)
            cfg["evaluator"](model, processor, dataset, image_dir, out_dir, ds_name, decoding_strategy=DECODING_STRATEGY)

if __name__ == "__main__":
    main()
