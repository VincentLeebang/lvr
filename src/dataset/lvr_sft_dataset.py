import copy
import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info, get_video_info, llava_to_openai_lvr, pad_sequence
import numpy as np
from PIL import Image
from typing import List, Tuple
import math

class SupervisedDatasetLVR(Dataset):
    """Dataset for supervised fine-tuning LVR model."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDatasetLVR, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))

        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height
        self.video_resized_w = data_args.video_resized_width
        self.video_resized_h = data_args.video_resized_height
        self.fps = data_args.fps

        '''Deprecated as image masking works better'''
        # if "Qwen" in model_id:
        #     self.bbox_token_mapper_manual = QwenVLBboxTokenMapper(patch_size=14,spatial_merge_size=2)

    def __len__(self):
        return len(self.list_data_dict)
        


    def make_bbox_masks_rgb(
        self,
        pil_imgs: List[Image.Image], 
        bboxes_norm: List[Tuple[float, float, float, float]]
    ) -> List[np.ndarray]:
        """
        Create RGB binary masks for multiple PIL images based on normalized bounding boxes.

        Args:
            pil_imgs: list of PIL.Image instances.
            bboxes_norm: list of bounding boxes (x_min, y_min, x_max, y_max), normalized [0,1].

        Returns:
            List of NumPy arrays, each with shape (H, W, 3), dtype=uint8:
                - 1 inside bbox (for each RGB channel), 0 outside.
        """
        assert len(pil_imgs) == len(bboxes_norm), "Images and bboxes lists must be same length"
        masks_rgb = []

        for img, bbox in zip(pil_imgs, bboxes_norm):
            w, h = img.size
            x_min, y_min, x_max, y_max = bbox

            xmin = int(round(x_min * w))
            ymin = int(round(y_min * h))
            xmax = int(round(x_max * w))
            ymax = int(round(y_max * h))

            xmin, ymin = max(xmin, 0), max(ymin, 0)
            xmax, ymax = min(xmax, w), min(ymax, h)

            # Create a single-channel mask and broadcast it to RGB
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[ymin:ymax, xmin:xmax] = 1
            mask_rgb = np.stack([mask] * 3, axis=-1)  # shape = (H, W, 3)

            masks_rgb.append(mask_rgb)

        return masks_rgb
    
    def bbox_to_token_idxs(
        self,
        images: List[Image.Image], 
        bboxes: List[Tuple[float, float, float, float]]
    ) -> List[np.ndarray]:
        # each elem in image_masks is a fake image that is used to go through the qwen img processing 
        image_masks = self.make_bbox_masks_rgb(images,bboxes)

        lvr_token_idxs_list = []
        for image_mask in image_masks:
            image_masks_processed, _ = self.processor.image_processor._preprocess(
                image_masks,
                do_resize=False,
                do_rescale=False,
                do_normalize=False,
                patch_size=self.processor.image_processor.patch_size,
                temporal_patch_size=self.processor.image_processor.temporal_patch_size,
                merge_size=self.processor.image_processor.merge_size,
                do_convert_rgb=False,
            )
            idxs = np.where(np.any(image_masks_processed != 0, axis=1))[0]
            lvr_token_idxs_list.append(idxs)
        return lvr_token_idxs_list

    def bbox_to_token_idxs_manual(
            self,  
            images: List[Image.Image], 
            bboxes: List[Tuple[float, float, float, float]]) -> List[np.ndarray]:
            """
            Convert bounding box coordinates to visual token indices.
            
            Args:
                bbox: Bounding box coordinates [a, b, c, d]
                image_height: Height of the input image in pixels
                image_width: Width of the input image in pixels
                bbox_format: Format of bbox - "xyxy" (x1,y1,x2,y2) or "xywh" (x,y,w,h)
                return_grid_coords: If True, also return grid coordinates
                
            Returns:
                List of token indices, optionally with grid coordinates
            """
            # Setup dimensions for this specific image
            token_idx_list = []
            for img, bbox in zip(images, bboxes):
                '''
                    Attention: 
                    Even if the bbox is normalized here, it is possible to mess up the cords 
                    as QWEN img processing will resize the image if its beyond/below max/min pixels.
                    I dont wanna modify their official code for img processing tbh. So please keep in mind that
                    THE BBOXES ARE SUPPOSED TO BE NORMALIZED

                '''
                patch_size = self.processor.image_processor.patch_size
                image_width = img.width
                image_height = img.height

                grid_height = image_height // patch_size
                grid_width = image_width // patch_size

                token_grid_height = grid_height // self.processor.image_processor.temporal_patch_size
                token_grid_width = grid_width // self.processor.image_processor.temporal_patch_size


                x1, y1, x2, y2 = bbox
                if max(x1, y1, x2, y2) > 1.0:
                    x1 /= image_width
                    y1 /= image_height
                    x2 /= image_width
                    y2 /= image_height
                
                # Clamp coordinates to valid range
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(1, x2), min(1, y2)
                
                # Convert to token grid coordinates
                # Map from image coordinates to token grid coordinates
                token_x1 = int(x1 * token_grid_width)
                token_y1 = int(y1 * token_grid_height)
                token_x2 = min(int(math.ceil(x2 * token_grid_width)), token_grid_width)
                token_y2 = min(int(math.ceil(y2 * token_grid_height)), token_grid_height)
                
                # Ensure we have at least one token
                if token_x2 <= token_x1:
                    token_x2 = token_x1 + 1
                if token_y2 <= token_y1:
                    token_y2 = token_y1 + 1
                
                # Generate token indices and grid coordinates
                token_indices = []
                
                for y in range(token_y1, token_y2):
                    for x in range(token_x1, token_x2):
                        # Convert 2D grid position to 1D token index
                        token_idx = y * token_grid_width + x
                        token_indices.append(token_idx)
                token_idx_list.append(np.array(token_indices))
            
            return token_idx_list
    

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """This code currently assumes single image + multi/single bboxes"""

        sources = self.list_data_dict[i]

        is_video = False

        processor = self.processor
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            
            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            
            for image_file in image_files:
                if not os.path.exists(image_file):
                    if not image_file.startswith("http"):
                        image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, self.image_min_pixel, self.image_max_pixel, self.image_resized_w, self.image_resized_h))
        else:
            grid_key = None
            pixel_key = None
            images=None
            videos=None

        # extracting token indexes for processed images
        bboxes = sources['bboxes']
        '''
            BBox to token idx method 1: trivial mapping
            Deprecated due to the granularity
            Remember to scale the idxes by 4 if using these
        '''
        lvr_token_idxs_list_manual = self.bbox_to_token_idxs_manual(images,bboxes)

        # an alternative method using an image mask to do the same thing 
        # lvr_token_idxs_list = self.bbox_to_token_idxs(images,bboxes)

        sources = copy.deepcopy(llava_to_openai_lvr(sources['conversations'], is_video=is_video,lvr_token_idxs_list=lvr_token_idxs_list_manual))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # Qwen2-VL uses a default system message so I've added this.
        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        for _, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
            
            if DEFAULT_IMAGE_TOKEN in user_input:
                inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                

            elif DEFAULT_VIDEO_TOKEN in user_input:
                '''Not Implemented'''
                pass
                # if "Qwen2.5" in self.model_id:
                #     inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt', **video_kwargs)
                #     all_second_gird.extend(inputs["second_per_grid_ts"])
                # else:
                #     inputs = processor(text=[user_input], images=images, videos=videos, padding=False, do_resize=False, return_tensors='pt')
                # prompt_input_ids = inputs['input_ids']
                # all_pixel_values.append(inputs[pixel_key])
                # all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            # filling the response with bboxes
            
            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)
        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL does not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        # input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        attention_mask = (input_ids > -1000000).to(torch.long)

        lvr_tokens = []
        for item_img in lvr_token_idxs_list_manual:
            group_lst = []
            for group in item_img:
                group_lst.append(torch.tensor(group))

            lvr_tokens.append(group_lst)

        # lvr_token_idxs_list_manual_ = torch.tensor(np.array(lvr_token_idxs_list_manual))

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            lvr_tokens=lvr_tokens
        )

        if pixel_key and grid_key:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict[pixel_key] = pixel_values
            data_dict[grid_key] = image_thw

        if len(all_second_gird) > 0:
            second_gird = all_second_gird
            data_dict["second_per_grid_ts"] = second_gird
        
        return data_dict

class DataCollatorForSupervisedDatasetLVR(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_lvr_tokens = []

        for example in examples:
            keys = example.keys()
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        lvr_tokens = [example['lvr_tokens'] for example in examples]
        lvr_tokens_all_local_indices = [torch.tensor(idx) for group in lvr_tokens for idx in group]

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'lvr_tokens': lvr_tokens_all_local_indices
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            # pixel_values_raw = batch_pixel_values   # Now its a list of pixel values
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            # data_dict["pixel_values_raw"] = pixel_values_raw
            data_dict["image_grid_thw"] = image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    
def make_supervised_data_module_lvr(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDatasetLVR(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDatasetLVR(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)