import torch
import torch.nn as nn
from torch.nn import LayerNorm

class LVRHead(nn.Module):
    """
        The simplest mlp w/o up_proj
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.ln_q = LayerNorm(hidden_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x))
        return x


from transformers.activations import ACT2FN
class LVRHeadGLU(nn.Module):
    ''' 
        The Gated Liner Unit MLP
    '''
    def __init__(self, hidden_size, intermediate_size, hidden_act, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        # 11008 for 3b; 18944 for 7b; 27648 for 32b
        self.intermediate_size = intermediate_size  
        self.hidden_act = hidden_act
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[self.hidden_act]    #silu

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
    



# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2RMSNorm
# class LVRHeadRMS(nn.Module):
#     """
#         Modified Patch Merger from transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
#         This inherits from the mm projector of qwen 2.5 vl
#     """
#     def __init__(self, hidden_size: int) -> None:
#         super().__init__()
#         self.ln_q = Qwen2RMSNorm(hidden_size, eps=1e-6)
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.GELU(),
#             nn.Linear(hidden_size, hidden_size),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.mlp(self.ln_q(x))
#         return x