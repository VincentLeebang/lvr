


# transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py


import torch
import transformers

def replace_qwen_2_5_vl_patch_emb():
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed.forward = Qwen2_5_VisionPatchEmbedFp32Forward

def Qwen2_5_VisionPatchEmbedFp32Forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        # hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        with torch.autocast(device_type=hidden_states.device.type, dtype=torch.float32):
            hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.view(-1, self.embed_dim).to(target_dtype)
        return hidden_states
