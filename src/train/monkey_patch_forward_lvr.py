import torch
from typing import Optional, List, Union, Tuple
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
import numpy as np
import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
from transformers.utils import is_torchdynamo_compiling,TransformersKwargs
from transformers.processing_utils import Unpack
from src.constants import IGNORE_INDEX
import torch.distributed as dist

import torch.nn.functional as F


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def replace_qwen2_5_with_mixed_modality_forward_lvr(inference_mode=False,
                                                    coconut=True,
                                                    lvr_head=True,
                                                    mode_switch_loss=False,
                                                    latent_end_token=False,
                                                    rl = False):
    
    print("#"*42)
    if inference_mode:
        if lvr_head:
            print("Inference mode with Lvr_head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_inference
        else:
            print("Inference mode without Lvr_head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_inference
    elif rl:
        print("Activated stage 2 training!!!")
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_grpo
    else:
        if latent_end_token and lvr_head:
            print("Activated latent end token mode with LVR_Head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_with_latentEndToken
        elif latent_end_token and not lvr_head:
            print("Activated latent end token mode without LVR_Head!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_latentEndToken
        elif mode_switch_loss:
            print("Activated BCE mode swtich loss!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head_with_modeSwitchLoss
        elif lvr_head:
            print("Activated naive LVR with head mode!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr_with_head
        else:
            print("Activated naive LVR without head mode!!!")
            transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.forward = qwen2_5_mixed_modality_forward_lvr
    
    print("#"*42)


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
    loss_mode_switch: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    
    last_position_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    # next_pos_lvr:Optional[bool] = False


def  set_lvr_loss_fct(loss_lvr_fct: str):
    """
        Set the loss function for LVR.
        Args:
            loss_lvr_fct (str): The type of loss function to use for LVR.
        Returns:
            A loss function object.
    """
    if loss_lvr_fct == 'mse':
        return MSELoss()
    elif loss_lvr_fct == 'mae':
        return L1Loss()
    elif loss_lvr_fct == 'cosine':
        # Returns a loss function: 1 - cosine similarity
        def cosine_loss(x, y):
            return 1 - F.cosine_similarity(x, y, dim=-1).mean()
        return cosine_loss
    else:
        raise ValueError(f"Unsupported lvr_loss: {loss_lvr_fct}")

'''
    Coconut mode
    No LVR Head
'''
def qwen2_5_mixed_modality_forward_lvr(
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
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if lvr_mode_switch:
        # only happen during inference
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    '''Only necessary in training'''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:
            
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
    
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_lvr_token_indices = []

                for b, lvr_ids in enumerate(lvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_lvr_token_indices.append(lvr_ids + offset)
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
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
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
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
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    No LVR Head
'''
def qwen2_5_mixed_modality_forward_lvr_inference(
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
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    if last_position_hidden_state is not None:
        # only happen during inference
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]

    '''Only necessary in training'''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    # if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
    #     # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
    #     dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
    #     dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
    #     dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
    #     image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
    #     # Operates as maksed_scatter for the image tokens
    #     # However the values are all zeros so it dosen't affect the embeddings.
    #     # This could avoid deepspeed error when some batch only has texts.
    #     inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:
            
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
    
        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_lvr_token_indices = []

                for b, lvr_ids in enumerate(lvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_lvr_token_indices.append(lvr_ids + offset)
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )
                self.model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                if cache_position is not None:
                    delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                else:
                    delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
                position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
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
            **kwargs
        )

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
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
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )



'''
    Coconut mode;
    LVR head;
    Note that this forward function is used for inferencing all the LVR models with a LVR head
'''
def qwen2_5_mixed_modality_forward_lvr_with_head(
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
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  
            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_lvr_token_indices = []
                for b, lvr_ids in enumerate(lvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_lvr_token_indices.append(lvr_ids + offset)
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
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

    '''apply lvr_head in training mode'''
    if lvr_tokens is not None and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
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
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )





def qwen2_5_mixed_modality_forward_lvr_with_head_inference(
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
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_tokens_thw: Optional[List[torch.Tensor]] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
    **kwargs: Unpack[TransformersKwargs],
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens is not None:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  
            if isinstance(lvr_tokens,list):
                '''Exrtacting tokens from original image'''
                #  GLOBAL starting index in `image_embeds` of each image in the batch
                image_token_offsets = torch.cumsum(
                    F.pad(total_tokens, (1, 0)), dim=0
                )[:-1]  # shape [B], offset into image_embeds for each batch element

                global_lvr_token_indices = []
                for b, lvr_ids in enumerate(lvr_tokens):
                    # Convert local to global index
                    offset = image_token_offsets[b].item()
                    global_lvr_token_indices.append(lvr_ids + offset)
                global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

                # Step 3: Gather the selected visual embeddings
                selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

                # Step 4: Replace in input_embeds at the right batch and position
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            
            else:
                '''re-encode target area'''
                # Now lvr_tokens is pixel_values of the cropped targets
                selected_lvr_embeds = self.model.get_image_features(lvr_tokens, lvr_tokens_thw)
                selected_lvr_embeds = torch.cat(selected_lvr_embeds, dim=0)
                inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
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
        **kwargs
    )

    '''apply lvr_head in training mode'''
    if lvr_tokens is not None and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

            # selected_hidden_states = outputs.last_hidden_state[batch_indices, seq_positions_start]
            # lvr_head_output = self.lvr_head(selected_hidden_states)
            # outputs.last_hidden_state[batch_indices, seq_positions_start] = lvr_head_output

    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
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
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    LVR head
'''
def qwen2_5_mixed_modality_forward_lvr_with_head_with_modeSwitchLoss(
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
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if lvr_mode_switch:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

        #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            global_lvr_token_indices = []

            for b, lvr_ids in enumerate(lvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_lvr_token_indices.append(lvr_ids + offset)
            global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
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

    '''apply lvr_head in training mode'''
    if lvr_tokens and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
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
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1  # Now points to lvr_start
        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

        # mode switch loss

        lvr_or_lvrstart_mask = (input_ids == self.config.lvr_start_id) | (input_ids == self.config.lvr_id)

        # Find the next tokens of each position
        shifted_input_ids = torch.roll(input_ids, shifts=-1, dims=1)
        # the lvr token that is right before lvr_end token
        is_last_lvr = lvr_or_lvrstart_mask & (shifted_input_ids == self.config.lvr_end_id)
        # 1 if it's the last <lvr> before <lvr_end>, else 0
        targets = is_last_lvr.float()  # [batch_size, seq_len]

        lvr_end_logits = logits[..., self.config.lvr_end_id]  # [batch_size, seq_len]

        # Apply mask to focus only on <lvr_start>,<lvr> token positions
        masked_logits = lvr_end_logits[lvr_or_lvrstart_mask]  # [num_lvr_tokens]
        masked_targets = targets[lvr_or_lvrstart_mask]        # [num_lvr_tokens]

        loss_mode_switch = F.binary_cross_entropy_with_logits(masked_logits, masked_targets)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


'''
    Coconut mode
    LVR Head
    Padded <LVR_end> latent token as the mode switching signal
'''
def qwen2_5_mixed_modality_forward_lvr_with_head_with_latentEndToken(
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
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if lvr_mode_switch:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

        #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            global_lvr_token_indices = []

            for b, lvr_ids in enumerate(lvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_lvr_token_indices.append(lvr_ids + offset)
            global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            '''Apply lvr_latent_end_token'''
            lvr_latent_end_mask = (input_ids == self.config.lvr_latent_end_id)
            batch_indices_latentend, seq_positions_latentend = torch.nonzero(lvr_latent_end_mask, as_tuple=True)
            if lvr_latent_end_mask.any():
                inputs_embeds[lvr_latent_end_mask] = self.lvr_latent_end_emb.to(inputs_embeds.device)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
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

    '''apply lvr_head in training mode'''
    if lvr_tokens and lvr_mask.any():
        # batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)
        if len(batch_indices) > 0:
            # Get last hidden states for <lvr> token positions, starting <lvr_start>
            seq_positions_start = seq_positions - 1  # shift left by 1 pos, now points to lvr_start
            outputs.last_hidden_state[batch_indices, seq_positions_start] = self.lvr_head(outputs.last_hidden_state[batch_indices, seq_positions_start])

            '''In this mode, <|lvr_latent_end|> is also a latent token'''
            seq_positions_start_latentend = seq_positions_latentend - 1
            outputs.last_hidden_state[batch_indices_latentend, seq_positions_start_latentend] = self.lvr_head(outputs.last_hidden_state[batch_indices_latentend, seq_positions_start_latentend])


    '''apply lvr_head in _inference mode'''
    if lvr_mode_switch:
        outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)
    mode_switch_loss_fct = set_lvr_loss_fct(self.config.loss_mode_switch_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
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
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill((shift_labels == self.config.lvr_id)|(shift_labels == self.config.lvr_latent_end_id), IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        # Get last hidden states for <lvr_latent_end> token positions
        seq_positions_start_latentend = seq_positions_latentend - 1
        selected_hidden_states_latentend = hidden_states[batch_indices_latentend, seq_positions_start_latentend].to(torch.float32)  # [L_total, H]

        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        selected_lvr_embeds_latentend = self.lvr_latent_end_emb.unsqueeze(0).expand_as(selected_hidden_states_latentend).to(torch.float32)
        selected_lvr_embeds_latentend = selected_lvr_embeds_latentend.to(selected_hidden_states_latentend.device)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds) 
        loss_mode_switch = mode_switch_loss_fct(selected_hidden_states_latentend, selected_lvr_embeds_latentend)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )



'''
    Coconut mode
    LVR Head
    Padded <LVR_end> latent token as the mode switching signal
'''
def qwen2_5_mixed_modality_forward_lvr_with_latentEndToken(
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
    lvr_tokens: Optional[torch.Tensor] = None,      # This is for TRAINING: Where should the lvr img tokens be
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if lvr_mode_switch:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)

        # IN TRAINING should we fill the lvr token positions with selected img tokrnd
        if lvr_tokens:
            '''
                Filling the lvr tokens with image embeddings.
                Applicable when each image input has multiple bboxes
            '''
            total_tokens = torch.sum(image_mask, dim=1)   # 1d tensor([216, 234, 234, 234]) for #vis_tokens in each instance in batch
            batch_size = input_ids.size(0) 
            # lvr mask for lvr token locations in the batch, [bs, seq_length]
            # in each instance, lvr tokens are True, others are False
            lvr_mask = input_ids == self.config.lvr_id  
            # Total length = number of <lvr> tokens in the batch
            # seq_positions: flattend LOCAL positions of lvr tokens in the inputs_ids
            batch_indices, seq_positions = torch.nonzero(lvr_mask, as_tuple=True)  

        #  GLOBAL starting index in `image_embeds` of each image in the batch
            image_token_offsets = torch.cumsum(
                F.pad(total_tokens, (1, 0)), dim=0
            )[:-1]  # shape [B], offset into image_embeds for each batch element

            global_lvr_token_indices = []

            for b, lvr_ids in enumerate(lvr_tokens):
                # Convert local to global index
                offset = image_token_offsets[b].item()
                global_lvr_token_indices.append(lvr_ids + offset)
            global_lvr_token_indices = torch.cat(global_lvr_token_indices, dim=0)  # [L_total]

            # Step 3: Gather the selected visual embeddings
            selected_lvr_embeds = image_embeds[global_lvr_token_indices]  # [L_total, H]

            # Step 4: Replace in input_embeds at the right batch and position
            inputs_embeds[batch_indices, seq_positions] = selected_lvr_embeds

            '''Apply lvr_latent_end_token'''
            lvr_latent_end_mask = (input_ids == self.config.lvr_latent_end_id)
            batch_indices_latentend, seq_positions_latentend = torch.nonzero(lvr_latent_end_mask, as_tuple=True)
            if lvr_latent_end_mask.any():
                inputs_embeds[lvr_latent_end_mask] = self.lvr_latent_end_emb.to(inputs_embeds.device)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

    outputs = self.model.language_model(
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
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)
    mode_switch_loss_fct = set_lvr_loss_fct(self.config.loss_mode_switch_fct)

    loss = None
    loss_ce = None
    loss_lvr = None
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
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill((shift_labels == self.config.lvr_id)|(shift_labels == self.config.lvr_latent_end_id), IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # lvr loss
        # Get last hidden states for <lvr> token positions
        seq_positions_start = seq_positions - 1
        selected_hidden_states = hidden_states[batch_indices, seq_positions_start].to(torch.float32)  # [L_total, H]
        # Get last hidden states for <lvr_latent_end> token positions
        seq_positions_start_latentend = seq_positions_latentend - 1
        selected_hidden_states_latentend = hidden_states[batch_indices_latentend, seq_positions_start_latentend].to(torch.float32)  # [L_total, H]

        ''' We need to convert to fp32 to avoid overflow by mse'''
        selected_lvr_embeds = selected_lvr_embeds.to(torch.float32)
        selected_lvr_embeds_latentend = self.lvr_latent_end_emb.unsqueeze(0).expand_as(selected_hidden_states_latentend).to(torch.float32)
        selected_lvr_embeds_latentend = selected_lvr_embeds_latentend.to(selected_hidden_states_latentend.device)
        # Compute LVR loss between predicted and inserted lvr embeddings
        loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds) 
        loss_mode_switch = mode_switch_loss_fct(selected_hidden_states_latentend, selected_lvr_embeds_latentend)


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        loss_mode_switch=loss_mode_switch,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )


"""
    Forward function for stage 2 RL
    Kinda messy since in this stage, the transofmers will be 4.51.3 < 4.54 in stage I
    Will fix this inconsistency in final release
"""
def qwen2_5_mixed_modality_forward_lvr_rl(
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
    lvr_mode_switch: Optional[torch.Tensor] = None, # This is for INFERENCE: Which instance in the batch is in lvr mode
    last_position_hidden_state: Optional[torch.FloatTensor] = None, # This is for INFERENCE: last hidden state of the last position
) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    '''In this mode, no lvr_tokens'''
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    

    if inputs_embeds is None:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

    ''' 
        only happen during inference 
        inputs_embeds in shape (bs, seq_len, hidden)
    '''
    if last_position_hidden_state is not None:
        # in fact, each instance's seq_len will be 1 in inference
        inputs_embeds[lvr_mode_switch,-1,:] = last_position_hidden_state[lvr_mode_switch]
    
    ''' Only necessary in training '''
    # Pass dummy image and dummy grid to the visual model to avoid deepspeed error.
    if not lvr_mode_switch and (pixel_values is None and pixel_values_videos is None):
        # Create dummy pixel_values and grid_thw for avoiding deepspeed error.
        dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
        dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
        
        dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
        image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
        # Operates as maksed_scatter for the image tokens
        # However the values are all zeros so it dosen't affect the embeddings.
        # This could avoid deepspeed error when some batch only has texts.
        inputs_embeds += image_embeds.mean() * 0
            
    if pixel_values is not None:

        # with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float32):
        #     # Ensure vision tower inputs are float32
        #     pixel_values = pixel_values.to(torch.float32) 
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)


        n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        if input_ids is None:
            image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            image_mask = image_mask.all(-1)
        else:
            image_mask = input_ids == self.config.image_token_id


        n_image_tokens = (image_mask).sum()
        image_mask_unsqueeze = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_embeds.shape[0]
        if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_unsqueeze, image_embeds)
            

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    if position_ids is None:
        # Calculate RoPE index once per generation in the pre-fill stage only.
        # When compiling, we can't check tensor values thus we check only input length
        # It is safe to assume that `length!=1` means we're in pre-fill because compiled
        # models currently cannot do asssisted decoding
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.model.rope_deltas is None:
            position_ids, rope_deltas = self.model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.model.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
            if cache_position is not None:
                delta = (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
            else:
                delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
            position_ids += delta.to(position_ids.device)

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

    # check if there is lvr_head
    if self.config.lvr_head:
        '''apply lvr_head in _inference mode'''
        if lvr_mode_switch is not None:
            outputs.last_hidden_state[lvr_mode_switch,:,:] = self.lvr_head(outputs.last_hidden_state[lvr_mode_switch,:,:])

    hidden_states = outputs[0]
    last_position_hidden_state = outputs.last_hidden_state[:,-1,:]
    logits = self.lm_head(hidden_states)

    loss = None
    loss_ce = None
    loss_lvr = None
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
        # Don't want CE loss for <lvr> token
        shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss_ce = loss_fct(shift_logits, shift_labels)

        # No lvr loss in this mode
        loss_lvr = None


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        # loss=loss,
        loss_ce=loss_ce,
        loss_lvr=loss_lvr,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.model.rope_deltas,
        last_position_hidden_state =last_position_hidden_state
    )







'''Liger kernel'''
# def qwen2_5_mixed_modality_forward_lvr_with_flce(
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
#             dummy_pixel = torch.zeros(784, 1176).to(self.model.visual.device)
#             dummy_grid = torch.tensor([[1, 28, 28]]).to(self.model.visual.device)
            
#             dummy_pixel = dummy_pixel.type(self.model.visual.dtype)
#             image_embeds = self.model.visual(dummy_pixel, grid_thw=dummy_grid)
#             # Operates as maksed_scatter for the image tokens
#             # However the values are all zeros so it dosen't affect the embeddings.
#             # This could avoid deepspeed error when some batch only has texts.
#             inputs_embeds += image_embeds.mean() * 0
            
#         if pixel_values is not None:
#             pixel_values = pixel_values.type(self.model.visual.dtype)
#             image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
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
#             pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
#             video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
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

#     lvr_loss_fct = set_lvr_loss_fct(self.config.loss_lvr_fct)


#     loss = None
#     loss_ce = None
#     loss_lvr = None
#     logits = None

#     if self.training and (labels is not None):
#         shift_hidden_states = hidden_states[..., :-1, :].contiguous()
#         shift_labels = labels[..., 1:].contiguous()

#         # Flatten tokens
#         shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
#         shift_labels = shift_labels.view(-1)
#         # Don't want CE loss for <lvr> token
#         shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)

#         lce = LigerFusedLinearCrossEntropyLoss()
#         loss_ce = lce(self.lm_head.weight, shift_hidden_states, shift_labels)

        
#         # lvr loss
#         # Get last hidden states for <lvr> token positions
#         seq_positions_start = seq_positions - 1  # Now points to lvr_start
#         selected_hidden_states = hidden_states[batch_indices, seq_positions_start]  # [L_total, H]
#         # Compute LVR loss between predicted and inserted lvr embeddings
#         loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)
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
#             # Don't want CE loss for <lvr> token
#             shift_labels = shift_labels.masked_fill(shift_labels == self.config.lvr_id, IGNORE_INDEX)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss_ce = loss_fct(shift_logits, shift_labels)

#             # lvr loss
#             # Get last hidden states for <lvr> token positions
#             seq_positions_start = seq_positions - 1  # Now points to lvr_start
#             selected_hidden_states = hidden_states[batch_indices, seq_positions_start]  # [L_total, H]
#             # Compute LVR loss between predicted and inserted lvr embeddings
#             loss_lvr = lvr_loss_fct(selected_hidden_states, selected_lvr_embeds)

#     if not return_dict:
#         output = (logits,) + outputs[1:]
#         return (loss,) + output if loss is not None else output

#     return Qwen2_5_VLCausalLMOutputWithPast(
#         loss=loss,
#         loss_ce=loss_ce,
#         loss_lvr=loss_lvr,
#         past_key_values=outputs.past_key_values,
#         hidden_states=outputs.hidden_states,
#         attentions=outputs.attentions,
#         rope_deltas=self.rope_deltas,
#     )
