
import re
import torch

from qwen_vl_utils import process_vision_info

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
    LVR_START_TOKEN,
    LVR_END_TOKEN,
    LVR_TOKEN,
    LVR_LATENT_END_TOKEN,
    LVR_PLACEHOLDER,
)


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = (
            VISION_START_TOKEN
            + DEFAULT_VIDEO_TOKEN
            + VISION_END_TOKEN
        )
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = (
            VISION_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + VISION_END_TOKEN
        )

    return re.sub(pattern, replacement, input_string)


def replace_lvr_tokens(
    input_string,
    lvr_token_idxs_list,
    latent_end_token,
    fixed_num_of_lvr_tokens
):
    """
    Replace each LVR placeholder with tokens for one bounding-box region.

    Option A:
    One placeholder corresponds to exactly one bounding-box group.
    """

    placeholder_count = input_string.count(LVR_PLACEHOLDER)

    # No LVR placeholder: preserve the original text.
    if placeholder_count == 0:
        return input_string

    # Fixed-token mode: each placeholder receives a fixed number of tokens.
    if fixed_num_of_lvr_tokens is not None:

        if fixed_num_of_lvr_tokens < 0:
            raise ValueError(
                "fixed_num_of_lvr_tokens cannot be negative."
            )

        replacement = (
            LVR_START_TOKEN
            + LVR_TOKEN * fixed_num_of_lvr_tokens
            + LVR_END_TOKEN
        )

        return input_string.replace(LVR_PLACEHOLDER, replacement)

    # Normal mode requires bounding-box groups.
    if lvr_token_idxs_list is None:
        raise ValueError(
            "LVR placeholders exist, but lvr_token_idxs_list is None."
        )

    # Every placeholder must have exactly one corresponding bbox group.
    if placeholder_count != len(lvr_token_idxs_list):
        raise ValueError(
            f"LVR mapping mismatch: found {placeholder_count} "
            f"placeholders but received {len(lvr_token_idxs_list)} "
            f"bounding-box groups."
        )

    # Split while preserving all text surrounding the placeholders.
    segments = input_string.split(LVR_PLACEHOLDER)

    result = segments[0]

    for idxs, following_text in zip(
        lvr_token_idxs_list,
        segments[1:]
    ):

        replacement = (
            LVR_START_TOKEN
            + LVR_TOKEN * len(idxs)
        )

        if latent_end_token:
            replacement += LVR_LATENT_END_TOKEN

        replacement += LVR_END_TOKEN

        result += replacement + following_text

    return result


def llava_to_openai_lvr(
    conversations,
    is_video=False,
    lvr_token_idxs_list=None,
    latent_end_token=False,
    fixed_num_of_lvr_tokens=None
):

    role_mapping = {
        "human": "user",
        "gpt": "assistant"
    }

    transformed_data = []

    for conversation in conversations:

        transformed_content = replace_image_tokens(
            conversation["value"],
            is_video=is_video
        )

        transformed_content = replace_lvr_tokens(
            transformed_content,
            lvr_token_idxs_list,
            latent_end_token,
            fixed_num_of_lvr_tokens
        )

        transformed_entry = {
            "role": role_mapping.get(
                conversation["from"],
                conversation["from"]
            ),
            "content": transformed_content,
        }

        transformed_data.append(transformed_entry)

    return transformed_data


def llava_to_openai(conversations, is_video=False):

    role_mapping = {
        "human": "user",
        "gpt": "assistant"
    }

    transformed_data = []

    for conversation in conversations:

        transformed_content = replace_image_tokens(
            conversation["value"],
            is_video=is_video
        )

        transformed_entry = {
            "role": role_mapping.get(
                conversation["from"],
                conversation["from"]
            ),
            "content": transformed_content,
        }

        transformed_data.append(transformed_entry)

    return transformed_data


def truncate_sequence(input_ids, labels, max_length, eos_token_id):

    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length - 1]
        labels = labels[:max_length - 1]

    if eos_token_id is not None:
        input_ids = torch.cat([
            input_ids,
            torch.tensor([eos_token_id])
        ])

        labels = torch.cat([
            labels,
            torch.tensor([eos_token_id])
        ])

    return input_ids, labels


def pad_sequence(
    sequences,
    padding_side='right',
    padding_value=0
):

    """
    Pad a list of tensors.

    sequences: list of tensors in [seq_len, *] shape
    """

    assert padding_side in ['right', 'left']

    max_size = sequences[0].size()
    trailing_dims = max_size[1:]

    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    output = sequences[0].new_full(
        (batch_size, max_len) + trailing_dims,
        padding_value
    )

    for i, seq in enumerate(sequences):

        length = seq.size(0)

        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq

    return output


def get_image_info(
    image_path,
    min_pixel,
    max_pixel,
    width,
    height
):

    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [
        {
            "role": "user",
            "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]


def get_video_info(
    video_path,
    min_pixels,
    max_pixels,
    width,
    height,
    fps
):

    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [
        {
            "role": "user",
            "content": [content]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True
    )

    return video_input[0], video_kwargs
