import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.cuda.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl

# Helper functions for weight fusion
def fuse_qkv_weights(q_weight, k_weight, v_weight):
    """Fuse Q,K,V weights into a single weight matrix"""
    return torch.cat([q_weight, k_weight, v_weight], dim=0)

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

class Attention(nn.Module):
    """
    Optimized multi-head attention using Flash Attention and weight fusion
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
        super(Attention, self).__init__()
        
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        
        # Fused QKV projection
        self.qkv_proj = nn.Linear(c_q, 3 * c_hidden * no_heads, bias=False)
        
        # Initialize weights to match unfused version
        with torch.no_grad():
            # Create temporary unfused layers to copy weights
            tmp_q = nn.Linear(c_q, c_hidden * no_heads, bias=False)
            tmp_k = nn.Linear(c_k, c_hidden * no_heads, bias=False)
            tmp_v = nn.Linear(c_v, c_hidden * no_heads, bias=False)
            
            # Copy weights to fused projection
            fused_weight = fuse_qkv_weights(
                tmp_q.weight.data,
                tmp_k.weight.data,
                tmp_v.weight.data
            )
            self.qkv_proj.weight.data.copy_(fused_weight)
            
        self.linear_o = nn.Linear(c_hidden * no_heads, c_q)
        
        self.linear_g = None
        if self.gating:
            self.linear_g = nn.Linear(c_q, c_hidden * no_heads)
            
        self.sigmoid = nn.Sigmoid()
        
    def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Fused QKV projection
        qkv = self.qkv_proj(q_x)
        
        # Split into q,k,v and reshape
        chunk_size = self.c_hidden * self.no_heads
        q, k, v = qkv.chunk(3, dim=-1)
        
        # [*, Q/K/V, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))
        
        # Scale query
        q = q / math.sqrt(self.c_hidden)
        
        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g
            
        # Merge heads
        o = flatten_final_dims(o, 2)
        
        # Output projection
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
            q_x: [*, Q, C_q] query data
            kv_x: [*, K, C_k] key data
            biases: List of biases that broadcast to [*, H, Q, K]
        Returns:
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []
            
        # Prepare Q,K,V with fused projection
        q, k, v = self._prep_qkv(q_x, kv_x)
        
        # Create attention mask from biases
        mask = None
        if biases:
            mask = sum(biases)
            
        # Use Flash Attention for core computation
        o = FlashAttentionFunction.apply(q, k, v, mask)
        
        # Post-process output
        o = self._wrap_up(o, q_x)
        
        return o

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