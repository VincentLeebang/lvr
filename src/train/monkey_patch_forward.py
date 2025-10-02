from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast
import torch
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import transformers.models.qwen2_vl.modeling_qwen2_vl
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
# from liger_kernel.transformers.fused_linear_cross_entropy import (
#     LigerFusedLinearCrossEntropyLoss
# )
from src.constants import IGNORE_INDEX

import torch.nn.functional as F

from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
        please refer to the original Qwen2_5_VLCausalLMOutputWithPast in transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
    """

    loss: Optional[torch.FloatTensor] = None
    loss_lvr: Optional[torch.FloatTensor] = None
    loss_ce: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

def replace_qwen_2_with_mixed_modality_forward(use_liger=True):
    # if use_liger:
    #     transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen_2_mixed_modality_forward_with_flce
    # else:
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward = qwen_2_mixed_modality_forward

def replace_qwen2_5_with_mixed_modality_forward(use_liger=True):
    # if use_liger:
    #     transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_with_flce
    # else:
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward

# def replace_qwen2_5_with_mixed_modality_forward_lvr(use_liger=False,coconut=False):
#     if coconut:
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_coconut
#     elif use_liger:
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_with_flce_lvr
#     else:
#         transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr

# def qwen_2_mixed_modality_forward_with_flce(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
# ):
    
#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)

#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(784, 1176).to(self.visual.get_device())
#             dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.get_device())
            
#             dummy_pixel = dummy_pixel.type(self.visual.get_dtype())
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0

#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.visual.get_dtype())
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )
#             image_mask = (
#                 (input_ids == self.config.image_token_id)
#                 .unsqueeze(-1)
#                 .expand_as(inputs_embeds)
#                 .to(inputs_embeds.device)
#             )
#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )
#             video_mask = (
#                 (input_ids == self.config.video_token_id)
#                 .unsqueeze(-1)
#                 .expand_as(inputs_embeds)
#                 .to(inputs_embeds.device)
#             )
#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids, image_grid_thw, video_grid_thw, attention_mask
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#     )

#     hidden_states = outputs[0]

#     loss = None
#     logits = None

#     if self.training and (labels is not None):
#         shift_hidden_states = hidden_states[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()

#         # Flatten tokens
#         shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
#         shift_labels = shift_labels.view(-1)

#         lce = LigerFusedLinearCrossEntropyLoss()
#         loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
#     else:
#         logits = self.lm_head(hidden_states)
#         if labels is not None:
#             # Upcast to float if we need to compute the loss to avoid potential precision issues
#             logits = logits.float()
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2VLCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )

def qwen_2_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
):
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)

        # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
        if pixel_values is None and pixel_values_videos is None:
            # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
            dummy_pixel = torch.zeros(784, 1176).to(self.visual.get_device())
            dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.get_device())
            
            dummy_pixel = dummy_pixel.type(self.visual.get_dtype())
            image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
            # Operates as maksed_scatter for the image tokens
            # However the values are all zeros so it dosen't affect the embeddings.
            # This could avoid deepspeed error when some batch only has texts.
            inputs_embeds += image_embeds.mean() * 0

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )

# def qwen2_5_mixed_modality_forward_with_flce(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
# ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)
    
#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
#             dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
            
#             dummy_pixel = dummy_pixel.type(self.visual.dtype)
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0
            
#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.visual.dtype)
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )

#             mask = input_ids == self.config.image_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             image_mask = mask_expanded.to(inputs_embeds.device)

#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )

#             mask = input_ids == self.config.video_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             video_mask = mask_expanded.to(inputs_embeds.device)

#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids,
#                 image_grid_thw,
#                 video_grid_thw,
#                 second_per_grid_ts,
#                 attention_mask,
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = (
#                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
#                 if cache_position is not None
#                 else 0
#             )
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#         cache_position=cache_position,
#     )

#     hidden_states = outputs[0]
    
#     loss = None
#     logits = None

#     if self.training and (labels is not None):
#         shift_hidden_states = hidden_states[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()

#         # Flatten tokens
#         shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
#         shift_labels = shift_labels.view(-1)

#         lce = LigerFusedLinearCrossEntropyLoss()
#         loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
#     else:
#         logits = self.lm_head(hidden_states)
#         if labels is not None:
#             # Upcast to float if we need to compute the loss to avoid potential precision issues
#             logits = logits.float()
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2_5_VLCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )

def qwen2_5_mixed_modality_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
    
        # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
        if pixel_values is None and pixel_values_videos is None:
            # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
            dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
            dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
            
            dummy_pixel = dummy_pixel.type(self.visual.dtype)
            image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
            # Operates as maksed_scatter for the image tokens
            # However the values are all zeros so it dosen't affect the embeddings.
            # This could avoid deepspeed error when some batch only has texts.
            inputs_embeds += image_embeds.mean() * 0
            
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )

# '''Working on this'''
# # forward function without using fused linear cross entropy
# def qwen2_5_mixed_modality_forward_lvr(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
#     lvr_tokens: Optional[torch.Tensor] = None,
# ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)
    
#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
#             dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
            
#             dummy_pixel = dummy_pixel.type(self.visual.dtype)
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0
            
#         if pixel_values is not None:

#             pixel_values = pixel_values.type(self.visual.dtype)
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )

#             mask = input_ids == self.config.image_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             image_mask = mask_expanded.to(inputs_embeds.device)

#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
#             '''
#                 Filling the lvr tokens with image embeddings.
#                 Applicable when each image input has multiple bboxes
#             '''
#             total_tokens = torch.sum(mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
#             batch_size = input_ids.size(0) 
#             # lvr mask for lvr token locations in the batch, [bs, seq_length]
#             # in each instance, lvr tokens are True, others are False
#             lvr_mask = input_ids == self.config.lvr_id  
#             # Total length = number of <lvr> tokens in the batch
#             # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
#             batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

#            #  GLOBAL starting index in `image_embeds` of each image in the batch
#             image_token_offsets = torch.cumsum(
#                 F.pad(total_tokens, (1, 0)), dim=0
#             )[:-1]  # shape [B], offset into image_embeds for each batch element

#             global_lvr_token_indices = []

#             for b, lvr_ids in enumerate(lvr_tokens):
#                 # Convert local to global index
#                 offset = image_token_offsets[b].item()
#                 global_lvr_token_indices.append(lvr_ids + offset)
#             global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

#             # Step 3: Gather the selected visual embeddings
#             selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

#             # Step 4: Replace in input_embeds at the right batch and position
#             # Prepare indexing
#             # replaced_embeds = inputs_embeds.clone()
#             inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )

#             mask = input_ids == self.config.video_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             video_mask = mask_expanded.to(inputs_embeds.device)

#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids,
#                 image_grid_thw,
#                 video_grid_thw,
#                 second_per_grid_ts,
#                 attention_mask,
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = (
#                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
#                 if cache_position is not None
#                 else 0
#             )
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#         cache_position=cache_position,
#     )

#     hidden_states = outputs[0]
#     logits = self.lm_head(hidden_states)

#     if self.config.loss_lvr_fct == 'mse':
#         lvr_loss_fct = MSELoss()
#     else:
#         raise ValueError(f"Unsupported lvr_loss: {self.lvr_loss}")

#     loss = None
#     loss_ce = None
#     loss_lvr = None
#     if labels is not None:
#         # Upcast to float if we need to compute the loss to avoid potential precision issues
#         logits = logits.float()
#         # Shift so that tokens < n predict n
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         # Flatten the tokens
#         loss_fct = CrossEntropyLoss()
#         shift_logits = shift_logits.view(-1, self.config.vocab_size)
#         shift_labels = shift_labels.view(-1)
#         # Don't want CE loss for <lvr> token
#         shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

#         # Enable model parallelism
#         shift_labels = shift_labels.to(shift_logits.device)
#         loss_ce = loss_fct(shift_logits, shift_labels)

#         # lvr loss
#         # Get last hidden states for <lvr> token positions
#         seq_positions_start = seq_positions - 1  # Now points to lvr_start
#         selected_hidden_states = hidden_states[batch_indices, seq_positions_start]  # [L_total, H]
#         # Compute LVR loss between predicted and inserted lvr embeddings
#         loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)


#         # Total loss = CE + LVR
#         # However, this loss is not used in training; the overwritten compute_loss function in QwenLVRSFTTrainer is used instead.
#         loss = loss_ce + loss_lvr


#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2_5_VLCausalLMOutputWithPast(
#         loss=loss,
#         loss_ce=loss_ce,
#         loss_lvr=loss_lvr,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )

# '''Undone'''
# def qwen2_5_mixed_modality_forward_with_flce_lvr(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
# ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)
    
#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
#             dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
            
#             dummy_pixel = dummy_pixel.type(self.visual.dtype)
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0
            
#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.visual.dtype)
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )

#             mask = input_ids == self.config.image_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             image_mask = mask_expanded.to(inputs_embeds.device)

#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )

#             mask = input_ids == self.config.video_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             video_mask = mask_expanded.to(inputs_embeds.device)

#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids,
#                 image_grid_thw,
#                 video_grid_thw,
#                 second_per_grid_ts,
#                 attention_mask,
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = (
#                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
#                 if cache_position is not None
#                 else 0
#             )
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#         cache_position=cache_position,
#     )

#     hidden_states = outputs[0]
    
#     loss = None
#     logits = None

#     if self.training and (labels is not None):
#         shift_hidden_states = hidden_states[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()

#         # Flatten tokens
#         shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
#         shift_labels = shift_labels.view(-1)

#         lce = LigerFusedLinearCrossEntropyLoss()
#         loss = lce(self.lm_head.weight, shift_hidden_states, shift_labels)
#     else:
#         logits = self.lm_head(hidden_states)
#         if labels is not None:
#             # Upcast to float if we need to compute the loss to avoid potential precision issues
#             logits = logits.float()
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2_5_VLCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )

# '''Undone'''
# def qwen2_5_mixed_modality_forward_lvr_coconut(
#     self,
#     input_ids: torch.LongTensor = None,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.LongTensor] = None,
#     past_key_values: Optional[List[torch.FloatTensor]] = None,
#     inputs_embeds: Optional[torch.FloatTensor] = None,
#     labels: Optional[torch.LongTensor] = None,
#     use_cache: Optional[bool] = None,
#     output_attentions: Optional[bool] = None,
#     output_hidden_states: Optional[bool] = None,
#     return_dict: Optional[bool] = None,
#     pixel_values: Optional[torch.Tensor] = None,
#     pixel_values_videos: Optional[torch.FloatTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     rope_deltas: Optional[torch.LongTensor] = None,
#     cache_position: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
# ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

#     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#     output_hidden_states = (
#         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#     )
#     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#     if inputs_embeds is None:
#         inputs_embeds = self.model.embed_tokens(input_ids)
    
#         # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
#         if pixel_values is None and pixel_values_videos is None:
#             # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
#             dummy_pixel = torch.zeros(784, 1176).to(self.visual.device)
#             dummy_grid = torch.tensor([[1, 28, 28]]).to(self.visual.device)
            
#             dummy_pixel = dummy_pixel.type(self.visual.dtype)
#             image_embeds = self.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0
            
#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.visual.dtype)
#             image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
#             n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
#             n_image_features = image_embeds.shape[0]
#             if n_image_tokens != n_image_features:
#                 raise ValueError(
#                     f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
#                 )

#             mask = input_ids == self.config.image_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             image_mask = mask_expanded.to(inputs_embeds.device)

#             image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

#         if pixel_values_videos is not None:
#             pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
#             video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
#             n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
#             n_video_features = video_embeds.shape[0]
#             if n_video_tokens != n_video_features:
#                 raise ValueError(
#                     f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
#                 )

#             mask = input_ids == self.config.video_token_id
#             mask_unsqueezed = mask.unsqueeze(-1)
#             mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
#             video_mask = mask_expanded.to(inputs_embeds.device)

#             video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
#             inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

#         if attention_mask is not None:
#             attention_mask = attention_mask.to(inputs_embeds.device)

#     # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
#     if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
#         # calculate RoPE index once per generation in the pre-fill stage only
#         if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
#             position_ids, rope_deltas = self.get_rope_index(
#                 input_ids,
#                 image_grid_thw,
#                 video_grid_thw,
#                 second_per_grid_ts,
#                 attention_mask,
#             )
#             self.rope_deltas = rope_deltas
#         # then use the prev pre-calculated rope-deltas to get the correct position ids
#         else:
#             batch_size, seq_length, _ = inputs_embeds.shape
#             delta = (
#                 (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
#                 if cache_position is not None
#                 else 0
#             )
#             position_ids = torch.arange(seq_length, device=inputs_embeds.device)
#             position_ids = position_ids.view(1, -1).expand(batch_size, -1)
#             if cache_position is not None:  # otherwise `deltas` is an int `0`
#                 delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
#             position_ids = position_ids.add(delta)
#             position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

#     outputs = self.model(
#         input_ids=None,
#         position_ids=position_ids,
#         attention_mask=attention_mask,
#         past_key_values=past_key_values,
#         inputs_embeds=inputs_embeds,
#         use_cache=use_cache,
#         output_attentions=output_attentions,
#         output_hidden_states=output_hidden_states,
#         return_dict=return_dict,
#         cache_position=cache_position,
#     )

#     hidden_states = outputs[0]
#     logits = self.lm_head(hidden_states)

#     loss = None
#     if labels is not None:
#         # Upcast to float if we need to compute the loss to avoid potential precision issues
#         logits = logits.float()
#         # Shift so that tokens < n predict n
#         shift_logits = logits[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()
#         # Flatten the tokens
#         loss_fct = CrossEntropyLoss()
#         shift_logits = shift_logits.view(-1, self.config.vocab_size)
#         shift_labels = shift_labels.view(-1)
#         # Enable model parallelism
#         shift_labels = shift_labels.to(shift_logits.device)
#         loss = loss_fct(shift_logits, shift_labels)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2_5_VLCausalLMOutputWithPast(
#         loss=loss,
#         logits=logits,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )
