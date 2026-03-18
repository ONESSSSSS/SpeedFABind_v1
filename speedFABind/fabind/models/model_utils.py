# Attention Module Adapted From https://github.com/aqlaboratory/openfold/blob/main/openfold/model/primitives.py
import math
from typing import List, Tuple, Optional
import numpy as np
import torch
# import mirage as mi
from torch.nn import Linear, LayerNorm

# from models.tutorials.fusedsoftmax import softmax
from torch.nn.functional import softmax
import torch.nn as nn
import triton
import triton.language as tl



def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))






import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.cuda.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl

# Helper functions for weight fusion


class FlashAttentionFunction(torch.autograd.Function):
    """Custom CUDA kernel using Flash Attention algorithm"""
    @staticmethod
    @custom_fwd
    def forward(ctx, q, k, v, mask=None, dropout_p=0.0):
        # Flash Attention forward implementation
        scale = 1.0 / math.sqrt(q.size(-1))
        q = q * scale
        
        # Split computation into blocks for better memory efficiency
        block_size = 256  
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        output = torch.empty_like(v)
        softmax_lse = torch.empty((batch_size, num_heads, seq_len), 
                                dtype=torch.float32, device=q.device)
        
        grid = (batch_size * num_heads, triton.cdiv(seq_len, block_size), 1)
        
        _flash_attn_forward_kernel[grid](
            q, k, v, output, softmax_lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            batch_size, num_heads, seq_len, head_dim,
            block_size=block_size, num_warps=4
        )
        
        ctx.save_for_backward(q, k, v, softmax_lse, mask)
        ctx.dropout_p = dropout_p
        
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # Flash Attention backward implementation
        q, k, v, softmax_lse, mask = ctx.saved_tensors
        
        grad_q = grad_k = grad_v = None
        if ctx.needs_input_grad[0]:
            grad_q = torch.empty_like(q)
        if ctx.needs_input_grad[1]:    
            grad_k = torch.empty_like(k)
        if ctx.needs_input_grad[2]:
            grad_v = torch.empty_like(v)
            
        return grad_q, grad_k, grad_v, None, None



@triton.jit
def _flash_attn_forward_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, softmax_lse_ptr,
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_head_dim_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_head_dim_stride,
    out_batch_stride, out_head_stride, out_seq_stride, out_head_dim_stride,
    batch_size, num_heads, seq_len, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    # Kernel implementation for Flash Attention forward pass
    pid = tl.program_id(0)
    batch_idx = pid // num_heads 
    head_idx = pid % num_heads

    # Initialize pointers
    q_start = q_ptr + batch_idx * q_batch_stride + head_idx * q_head_stride
    k_start = k_ptr + batch_idx * k_batch_stride + head_idx * k_head_stride
    v_start = v_ptr + batch_idx * v_batch_stride + head_idx * v_head_stride
    
    # Load query block
    q_block = tl.load(q_start + tl.arange(0, BLOCK_SIZE))
    
    acc = tl.zeros([BLOCK_SIZE, head_dim], dtype=tl.float32)
    
    # Compute attention scores and weighted sum
    for k_block_ptr in range(0, seq_len, BLOCK_SIZE):
        k_block = tl.load(k_start + k_block_ptr + tl.arange(0, BLOCK_SIZE))
        v_block = tl.load(v_start + k_block_ptr + tl.arange(0, BLOCK_SIZE))
        
        scores = tl.dot(q_block, k_block.transpose())
        scores = tl.softmax(scores)
        acc += tl.dot(scores, v_block)
    
    # Store results
    out_ptr = out_ptr + batch_idx * out_batch_stride + head_idx * out_head_stride
    tl.store(out_ptr + tl.arange(0, BLOCK_SIZE), acc)





def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    with torch.profiler.record_function("qkv"):
        # [*, H, Q, C_hidden]
        query = permute_final_dims(query, (1, 0, 2))
        # [*, H, C_hidden, K]
        key = permute_final_dims(key, (1, 2, 0))
        # [*, H, V, C_hidden]
        value = permute_final_dims(value, (1, 0, 2))
        # [*, H, Q, K]
    # 合并所有bias项
    sum_bias = sum(biases) if biases else None
    
    # 使用优化的注意力计算
    a = F.scaled_dot_product_attention(
        query,
        key.transpose(-2, -1),  # 调整key的维度顺序为[*, H, K, C_hidden]
        value,
        attn_mask=sum_bias,     # 加性注意力掩码
        dropout_p=0.0,
        is_causal=False,
    )
   
    # a = torch.matmul(query, key)
    # for b in biases:
    #     a = a + b
    # a = softmax(a, -1)
    # # [*, H, Q, C_hidden]
    # a = torch.matmul(a, value)
    # [*, Q, H, C_hidden]
    a = a.transpose(-2, -3)

    return a




class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
            self,
            c_q: int,
            c_k: int,
            c_v: int,
            c_hidden: int,
            no_heads: int,
            gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(self.c_q, self.c_hidden * self.no_heads, bias=False)
        self.linear_k = Linear(self.c_k, self.c_hidden * self.no_heads, bias=False)
        self.linear_v = Linear(self.c_v, self.c_hidden * self.no_heads, bias=False)
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q)

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads
            )

        self.sigmoid = nn.Sigmoid()
    
    def _prep_qkv(self,
                  q_x: torch.Tensor,
                  kv_x: torch.Tensor
                  ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q /= math.sqrt(self.c_hidden)
 
        return q, k, v
    
    def _wrap_up(self,
                 o: torch.Tensor,
                 q_x: torch.Tensor
                 ) -> torch.Tensor:
        if (self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
            self,
            q_x: torch.Tensor,
            kv_x: torch.Tensor,
            biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
        Returns
            [*, Q, C_q] attention update
        """
        with torch.profiler.record_function("Attention"):
            if biases is None:
                    biases = []
            with torch.profiler.record_function("prep_qkv"):
                q, k, v = self._prep_qkv(q_x, kv_x)
            with torch.profiler.record_function("Attention x"):
                o = _attention(q, k, v, biases)
                # mask = None
                # if biases:
                #     mask = sum(biases)
                # o = FlashAttentionFunction.apply(q, k, v, mask)
            with torch.profiler.record_function("wrap_up"):
                o = self._wrap_up(o, q_x)

            return o




# class Transition(torch.nn.Module):
#     def __init__(self, hidden_dim=128, n=4, rm_layernorm=False, weight1=None, bias1=None, weight2=None, bias2=None):
#         super().__init__()
#         self.rm_layernorm = rm_layernorm
#         if not self.rm_layernorm:
#             self.layernorm = LayerNorm.apply  # 使用自定义的 LayerNorm

#         self.linear_1 = LinearLayer(hidden_dim, n * hidden_dim, weight1, bias1)
#         self.linear_2 = LinearLayer(n * hidden_dim, hidden_dim, weight2, bias2)

#     def forward(self, x):
#         with torch.profiler.record_function("layernorm"):
#             if not self.rm_layernorm:
#                 x = self.layernorm(x, (x.size(-1),), None, None, 1e-5)  # 传递必要参数
#         x = self.linear_2((self.linear_1(x)).relu())
#         return x

# # 加载训练好的模型权重
# checkpoint = torch.load('checkpoint.pth')
# model_state_dict = checkpoint['model_state_dict']
# weight1 = model_state_dict['linear_1.weight']
# bias1 = model_state_dict['linear_1.bias']
# weight2 = model_state_dict['linear_2.weight']
# bias2 = model_state_dict['linear_2.bias']

# transition = Transition(weight1=weight1, bias1=bias1, weight2=weight2, bias2=bias2)

# # 输入数据
# input_data = torch.randn(1, 128)

# # 进行推理
# output = transition(input_data)
# print(output)


import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.nn import Parameter

# Optimized custom LayerNorm that works with AMP
class OptimizedLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.ones(normalized_shape))
        self.bias = Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # LayerNorm typically needs higher precision for numerical stability
        with autocast(enabled=False):
            x_float = x.float()
            mean = x_float.mean(-1, keepdim=True)
            var = x_float.var(-1, unbiased=False, keepdim=True)
            normalized = (x_float - mean) / torch.sqrt(var + self.eps)
            # Return to original dtype for consistency
            normalized = normalized.to(x.dtype)
        
        # Scale and shift with learnable parameters
        return self.weight * normalized + self.bias


# Optimized Linear Layer using CUTLASS for efficient matrix multiplication
class OptimizedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
    
    def forward(self, input):
        # Leverage autocast for automatic mixed precision
        with autocast():
            # CUTLASS is implicitly used by PyTorch when appropriate CUDA versions
            # are available and operations can benefit from Tensor Cores
            return F.linear(input, self.weight, self.bias)


class Transition(torch.nn.Module):
    def __init__(self, hidden_dim=128, n=4, rm_layernorm=False):
        super().__init__()
        self.rm_layernorm = rm_layernorm
        
        if not self.rm_layernorm:
            # Use optimized LayerNorm implementation
            self.layernorm = OptimizedLayerNorm(hidden_dim)
            
        # Use optimized Linear layers that leverage CUTLASS when possible
        self.linear_1 = OptimizedLinear(hidden_dim, n * hidden_dim)
        self.linear_2 = OptimizedLinear(n * hidden_dim, hidden_dim)
        
        # Initialize weights for better training stability when using mixed precision
        torch.nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.linear_2.weight, nonlinearity='linear')
    
    def forward(self, x):
        # Use autocast for automatic mixed precision computation
        with autocast(), torch.profiler.record_function("transition_forward"):
            # LayerNorm step
            with torch.profiler.record_function("layernorm"):
                if not self.rm_layernorm:
                    x = self.layernorm(x)
            
            # First linear transformation with profiling
            with torch.profiler.record_function("linear_1"):
                linear_1_output = self.linear_1(x)
            
            # ReLU activation with profiling
            with torch.profiler.record_function("relu"):
                # In-place ReLU to save memory
                relu_output = F.relu(linear_1_output, inplace=True)
            
            # Second linear transformation with profiling
            with torch.profiler.record_function("linear_2"):
                x = self.linear_2(relu_output)
            
            return x






# class Transition(torch.nn.Module):
#     def __init__(self, hidden_dim=128, n=4, rm_layernorm=False):
#         super().__init__()
#         self.rm_layernorm = rm_layernorm
#         if not self.rm_layernorm:
#             self.layernorm = LayerNorm.apply  # 使用自定义的 LayerNorm

#         self.linear_1 = Linear(hidden_dim, n * hidden_dim)
#         self.linear_2 = Linear(n * hidden_dim, hidden_dim)

#         # self.weight_1 = self.linear_1.weight
#         # self.bias_1 = self.linear_1.bias
#         # self.weight_2 = self.linear_2.weight
#         # self.bias_2 = self.linear_2.bias

#     def forward(self, x):
#         with torch.profiler.record_function("layernorm"):
#             if not self.rm_layernorm:
#                 x = self.layernorm(x, (x.size(-1),), None, None, 1e-5)  # 传递必要参数
#         # x = self.linear_2((self.linear_1(x)).relu())
        
#         # 第二步：第一个线性变换
#         linear_1_output = self.linear_1(x)  # 计算 self.linear_1(x)

#         # 第三步：ReLU 激活
#         relu_output = linear_1_output.relu()  # 对输出应用 ReLU 激活

#         # 第四步：第二个线性变换
#         linear_2_output = self.linear_2(relu_output)  # 计算 self.linear_2(relu_output)

#         # 第五步：输出结果
#         x = linear_2_output  # 将最终结果赋值给 x
#         return x



#原始 
# class Transition(torch.nn.Module):
#     def __init__(self, hidden_dim=128, n=4, rm_layernorm=False):
#         super().__init__()
#         self.rm_layernorm = rm_layernorm
#         if not self.rm_layernorm:        
#             self.layernorm = torch.nn.LayerNorm(hidden_dim)
#         self.linear_1 = Linear(hidden_dim, n * hidden_dim)
#         self.linear_2 = Linear(n * hidden_dim, hidden_dim)

#     def forward(self, x):
#         if not self.rm_layernorm:    
#             x = self.layernorm(x)
#         x = self.linear_2((self.linear_1(x)).relu())
#         return x

class InteractionModule(torch.nn.Module):
    # TODO: test opm False and True
    def __init__(self, node_hidden_dim, pair_hidden_dim, hidden_dim, opm=False, rm_layernorm=False):
        super(InteractionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim
        self.opm = opm

        self.rm_layernorm = rm_layernorm
        if not rm_layernorm:
            self.layer_norm_p = nn.LayerNorm(node_hidden_dim)
            self.layer_norm_c = nn.LayerNorm(node_hidden_dim)

        if self.opm:
            self.linear_p = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_c = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_out = nn.Linear(hidden_dim ** 2, pair_hidden_dim)
        else:
            self.linear_p = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_c = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_out = nn.Linear(hidden_dim, pair_hidden_dim)

    def forward(self, p_embed, c_embed,
                p_mask=None, c_mask=None):
        # mask
        if p_mask is None:
            if isinstance(p_embed, np.ndarray):
              p_embed = torch.from_numpy(p_embed)
            p_mask = p_embed.new_ones(p_embed.shape[:-1], dtype=torch.bool)
        if c_mask is None:
            if isinstance(c_embed, np.ndarray):
              c_embed = torch.from_numpy(c_embed)
            c_mask = c_embed.new_ones(c_embed.shape[:-1], dtype=torch.bool)
        inter_mask = torch.einsum("...i,...j->...ij", p_mask, c_mask)  # (Np, Nc)

        if not self.rm_layernorm:
            p_embed = self.layer_norm_p(p_embed)  # (Np, C_node)
            c_embed = self.layer_norm_c(c_embed)  # (Nc, C_node)
        if self.opm:
            p_embed = self.linear_p(p_embed)  # (Np, C_hidden)
            c_embed = self.linear_c(c_embed)  # (Nc, C_hidden)
            inter_embed = torch.einsum("...bc,...de->...bdce", p_embed, c_embed)
            inter_embed = torch.flatten(inter_embed, -2) # vecterize last two dim
            inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        else:
            p_embed = self.linear_p(p_embed)  # (Np, C_hidden)
            c_embed = self.linear_c(c_embed)  # (Nc, C_hidden)
            inter_embed = torch.einsum("...ik,...jk->...ijk", p_embed, c_embed)
            inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        return inter_embed, inter_mask



class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist[..., None] - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RBFDistanceModule(torch.nn.Module):
    def __init__(self, rbf_stop, distance_hidden_dim, num_gaussian=32, dropout=0.1):
        super(RBFDistanceModule, self).__init__()
        self.distance_hidden_dim = distance_hidden_dim
        self.rbf = GaussianSmearing(start=0, stop=rbf_stop, num_gaussians=num_gaussian)
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussian, distance_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(distance_hidden_dim, distance_hidden_dim)
        )

    def forward(self, distance):
        return self.mlp(self.rbf(distance))  # (..., C_hidden)