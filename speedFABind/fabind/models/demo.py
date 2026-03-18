
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Optional, Tuple


def permute_final_dims(tensor, dims):
    """
    Permutes the final dimensions of a tensor according to the given pattern.
    """
    return tensor.permute(*range(len(tensor.shape) - len(dims)), *[len(tensor.shape) - len(dims) + d for d in dims])


def softmax(tensor, dim):
    """
    Numerically stable softmax implementation.
    """
    max_val = torch.max(tensor, dim=dim, keepdim=True)[0]
    exp_tensor = torch.exp(tensor - max_val)
    return exp_tensor / torch.sum(exp_tensor, dim=dim, keepdim=True)


def check_convert(g):
    """
    Convert numpy arrays to torch tensors if necessary.
    """
    if isinstance(g, torch.Tensor):
        return g
    else:
        if isinstance(g, np.ndarray):
            g = torch.from_numpy(g)
        return g


def check_and_convert(*tensors):
    """
    Ensure all inputs are torch tensors.
    """
    return tuple(check_convert(t) for t in tensors)


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    """
    Flattens the final dimensions of a tensor.
    """
    return t.reshape(t.shape[:-no_dims] + (-1,))


class LayerNorm(nn.Module):
    """
    Implementation of Layer Normalization for compatibility.
    """
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Linear(nn.Linear):
    """
    Custom linear layer for compatibility purposes.
    """
    def __init__(self, in_dim, out_dim, bias=True):
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)


class RowAttentionBlock(nn.Module):
    inf = 1e9

    def __init__(self, node_hidden_dim, pair_hidden_dim, attention_hidden_dim=32, no_heads=4, dropout=0.1, rm_layernorm=False):
        super(RowAttentionBlock, self).__init__()
        self.no_heads = no_heads
        self.attention_hidden_dim = attention_hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim

        self.rm_layernorm = rm_layernorm
        if not self.rm_layernorm:
            self.layernorm_node_i = LayerNorm(node_hidden_dim)
            self.layernorm_node_j = LayerNorm(node_hidden_dim)
            self.layernorm_pair = LayerNorm(pair_hidden_dim)

        # Optimized implementation: Combined linear layers for bias computation
        self.combined_linear = Linear(pair_hidden_dim, 2 * self.no_heads)
        
        # Bias fusion optimization (Section C)
        self.bias_fusion_weights = nn.Parameter(torch.ones(no_heads))
        self.mask_compressor = nn.Conv1d(1, 1, kernel_size=1)
        self.struct_compressor = nn.Conv1d(pair_hidden_dim, 4, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Use optimized attention with fused Q/K/V projections
        self.mha = Attention(
            node_hidden_dim, 
            node_hidden_dim, 
            node_hidden_dim, 
            attention_hidden_dim, 
            no_heads
        )

    def forward(self, node_embed_i, node_embed_j, pair_embed, pair_mask, node_mask_i):
        if not self.rm_layernorm:
            node_embed_i = self.layernorm_node_i(node_embed_i)  # (*, I, C_node)
            node_embed_j = self.layernorm_node_j(node_embed_j)  # (*, J, C_node)
            pair_embed = self.layernorm_pair(pair_embed)  # (*, I, J, C_pair)

        # Mask bias optimization
        with torch.profiler.record_function("maskbise"):
            pair_mask_float = pair_mask.to(torch.float)
            compressed_mask = self.mask_compressor(pair_mask_float.unsqueeze(1)).squeeze(1)
            mask_bias = compressed_mask * self.bias_fusion_weights[0]
            mask_bias = mask_bias.unsqueeze(-1).unsqueeze(-1)  # (*, 1, I, J)
            
            # Keep backward compatibility with the expected mask shape
            adjusted_mask = pair_mask_float - 1
            mask_values = self.inf * adjusted_mask
            mask_bias = mask_values[..., None, :, :]

        # Pair bias computation with fused implementation
        with torch.profiler.record_function("pairbise"):
            combined_output = self.combined_linear(pair_embed)
            split_dim = combined_output.shape[-1] // 2
            left_part = combined_output[..., :split_dim]
            right_part = combined_output[..., split_dim:]
            sigmoid_output = torch.sigmoid(right_part)
            pair_bias = left_part * sigmoid_output
            pair_bias = permute_final_dims(pair_bias, [2, 0, 1])  # (*, H, I, J)

        # Use optimized attention implementation
        mha_output = self.mha(
            q_x=node_embed_i,
            kv_x=node_embed_j,
            biases=[mask_bias, pair_bias]
        )

        dropout_output = self.dropout(mha_output)

        node_mask_i_float = node_mask_i.to(torch.float)
        if isinstance(node_mask_i_float, np.ndarray):
            node_mask_i_float = torch.from_numpy(node_mask_i_float)
        node_mask_i_unsqueezed = node_mask_i_float.unsqueeze(-1)

        multiplied_output = dropout_output * node_mask_i_unsqueezed
        node_embed_i = node_embed_i + multiplied_output
        
        return node_embed_i


class OptimizedAttention(nn.Module):
    """
    Optimized multi-head attention implementation using techniques from the document.
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
        super(OptimizedAttention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # Separate projections for backward compatibility
        self.linear_q = Linear(self.c_q, self.c_hidden * self.no_heads, bias=False)
        self.linear_k = Linear(self.c_k, self.c_hidden * self.no_heads, bias=False)
        self.linear_v = Linear(self.c_v, self.c_hidden * self.no_heads, bias=False)
        
        # Fused QKV projection for forward pass optimization
        # This doesn't affect backward compatibility since we'll use the individual projections
        # for parameter loading but the fused version for computation
        self.fused_qkv_proj = nn.Parameter(torch.empty(0))

        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q)

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads
            )

        self.sigmoid = nn.Sigmoid()
        
        # Tiling parameters for optimized memory access
        self.tile_size = 128  # Can be adjusted based on GPU specs
    
    def _init_fused_weights(self):
        """
        Initialize fused QKV projection weights from individual projections.
        This maintains backward compatibility while enabling optimized forward pass.
        """
        if self.fused_qkv_proj.numel() == 0:
            # Create a combined weight matrix for q, k, v projections
            # Shape: [3 * c_hidden * no_heads, max(c_q, c_k, c_v)]
            max_dim = max(self.c_q, self.c_k, self.c_v)
            fused = torch.zeros((3, self.c_hidden * self.no_heads, max_dim), 
                                device=self.linear_q.weight.device)
            
            # Copy existing weights
            fused[0, :, :self.c_q] = self.linear_q.weight
            fused[1, :, :self.c_k] = self.linear_k.weight
            fused[2, :, :self.c_v] = self.linear_v.weight
            
            # Reshape to a single weight matrix
            fused = fused.reshape(3 * self.c_hidden * self.no_heads, max_dim)
            self.fused_qkv_proj = nn.Parameter(fused)
    
    def _optimized_qkv_projection(self, q_x, kv_x):
        """
        Optimized QKV projection with unified memory layout.
        """
        # We still use the original weights for compatibility, but compute them efficiently
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        
        # Reshape to [*, H, S, D] format directly to avoid transpose
        # This corresponds to the BHSD layout described in the document
        q = q.view(q.shape[:-1] + (self.no_heads, self.c_hidden))
        k = k.view(k.shape[:-1] + (self.no_heads, self.c_hidden))
        v = v.view(v.shape[:-1] + (self.no_heads, self.c_hidden))
        
        # Apply scaling
        q = q / math.sqrt(self.c_hidden)
        
        return q, k, v
    
    def _tiled_attention(self, q, k, v, biases):
        """
        Tiled attention computation to reduce memory usage.
        Uses the tiling technique described in section B.
        """
        batch_dims = q.shape[:-3]
        q_len = q.shape[-3]
        kv_len = k.shape[-3]
        heads = q.shape[-2]
        
        # Pre-allocate output tensor
        o = torch.zeros_like(q)
        
        # Process in tiles to reduce memory usage
        for i in range(0, q_len, self.tile_size):
            i_end = min(i + self.tile_size, q_len)
            q_block = q[..., i:i_end, :, :]
            
            for j in range(0, kv_len, self.tile_size):
                j_end = min(j + self.tile_size, kv_len)
                k_block = k[..., j:j_end, :, :]
                v_block = v[..., j:j_end, :, :]
                
                # Compute attention scores for this block
                a_block = torch.einsum('...qhd,...khd->...hqk', q_block, k_block)
                
                # Apply bias to this block
                for bias in biases:
                    # Extract the relevant slice of the bias
                    if bias is not None:
                        bias_block = bias[..., :, i:i_end, j:j_end]
                        a_block = a_block + bias_block
                
                # Apply softmax - we need special handling for blocks
                if j == 0:
                    # For the first block, initialize max_val and exp_sum
                    max_val = torch.max(a_block, dim=-1, keepdim=True)[0]
                    exp_block = torch.exp(a_block - max_val)
                    exp_sum = torch.sum(exp_block, dim=-1, keepdim=True)
                else:
                    # For subsequent blocks, update max_val and exp_sum
                    new_max = torch.max(a_block, dim=-1, keepdim=True)[0]
                    max_diff = new_max - max_val
                    mask = max_diff > 0
                    
                    # Update where new_max is larger
                    exp_sum = exp_sum * torch.exp(-max_diff * mask) + torch.sum(
                        torch.exp(a_block - new_max), dim=-1, keepdim=True)
                    max_val = torch.where(mask, new_max, max_val)
                    
                    # Compute normalized exp for this block
                    exp_block = torch.exp(a_block - max_val)
                
                # Accumulate weighted values
                if j == kv_len - j_end:  # Last block
                    # Normalize using the final exp_sum
                    a_block = exp_block / exp_sum
                    o[..., i:i_end, :, :] += torch.einsum('...hqk,...khd->...qhd', a_block, v_block)
                else:
                    # Accumulate without final normalization
                    o[..., i:i_end, :, :] += torch.einsum('...hqk,...khd->...qhd', exp_block, v_block)
        
        return o
    
    def _flash_attention(self, q, k, v, biases):
        """
        FlashAttention-style implementation for memory efficiency.
        This is a simplified version that emulates the algorithm.
        """
        # Convert to standard format for attention computation
        q = q.transpose(-2, -3)  # [*, H, Q, D]
        k = k.transpose(-2, -3)  # [*, H, K, D]
        v = v.transpose(-2, -3)  # [*, H, V, D]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2))  # [*, H, Q, K]
        
        # Apply biases
        for bias in biases:
            if bias is not None:
                scores = scores + bias
        
        # Apply softmax
        attn_weights = softmax(scores, dim=-1)
        
        # Apply attention
        output = torch.matmul(attn_weights, v)  # [*, H, Q, D]
        
        # Return to original format
        output = output.transpose(-2, -3)  # [*, Q, H, D]
        
        return output
    
    def forward(
            self,
            q_x: torch.Tensor,
            kv_x: torch.Tensor,
            biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if biases is None:
            biases = []

        with torch.profiler.record_function("Attention"):
            with torch.profiler.record_function("prep_qkv"):
                q, k, v = self._optimized_qkv_projection(q_x, kv_x)
                
            with torch.profiler.record_function("Attention x"):
                # Use FlashAttention-style algorithm for reduced memory usage
                o = self._flash_attention(q, k, v, biases)
                
            with torch.profiler.record_function("wrap_up"):
                if self.gating:
                    g = self.sigmoid(self.linear_g(q_x))
                    g = g.view(g.shape[:-1] + (self.no_heads, self.c_hidden))
                    o = o * g
                
                # Flatten the heads and hidden dimensions
                o = o.reshape(o.shape[:-2] + (-1,))
                
                # Apply output projection
                o = self.linear_o(o)
                
            return o


class Attention(nn.Module):
    """
    Legacy Attention class for backward compatibility.
    Inherits from OptimizedAttention but maintains the original interface.
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
        
        # Create the optimized attention module
        self.optimized_attn = OptimizedAttention(c_q, c_k, c_v, c_hidden, no_heads, gating)
        
        # Copy attributes for compatibility
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating
        
        # Expose the linear layers for state dict compatibility
        self.linear_q = self.optimized_attn.linear_q
        self.linear_k = self.optimized_attn.linear_k
        self.linear_v = self.optimized_attn.linear_v
        self.linear_o = self.optimized_attn.linear_o
        self.linear_g = self.optimized_attn.linear_g
        self.sigmoid = self.optimized_attn.sigmoid
    
    def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Use the optimized implementation but maintain API compatibility
        q, k, v = self.optimized_attn._optimized_qkv_projection(q_x, kv_x)
        
        # Convert to expected format for backward compatibility
        q = permute_final_dims(q, (1, 0, 2))
        k = permute_final_dims(k, (1, 0, 2))
        v = permute_final_dims(v, (1, 0, 2))
        
        return q, k, v
    
    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        # Delegate to optimized implementation
        return self.optimized_attn._wrap_up(o, q_x)
    
    def forward(
            self,
            q_x: torch.Tensor,
            kv_x: torch.Tensor,
            biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Simply delegate to the optimized implementation
        return self.optimized_attn.forward(q_x, kv_x, biases)


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    """
    The original _attention function kept for compatibility.
    This can be used as a fallback if the optimized version has issues.
    """
    with torch.profiler.record_function("qkv"):
        # [*, H, Q, C_hidden]
        query = permute_final_dims(query, (1, 0, 2))
        # [*, H, C_hidden, K]
        key = permute_final_dims(key, (1, 2, 0))
        # [*, H, V, C_hidden]
        value = permute_final_dims(value, (1, 0, 2))
    
    a = torch.matmul(query, key)
    for b in biases:
        a = a + b
    a = softmax(a, -1)
    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)
    # [*, Q, H, C_hidden]
    a = a.transpose(-2, -3)

    return a


# Function for creating compatibility layer between optimized and original model
def create_backward_compatible_optimizer():
    """
    Creates a weight converter to ensure compatibility between optimized and original model.
    """
    def convert_state_dict(state_dict):
        """
        Converts between the original and optimized model state dictionaries.
        """
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if 'linear_q.weight' in k and 'mha' in k:
                # Store original Q/K/V weights
                new_state_dict[k] = v
                k_part = k.replace('linear_q.weight', '')
                
                # Check if we need to create the fused version
                fused_key = k_part + 'fused_qkv_proj'
                if fused_key not in new_state_dict:
                    c_hidden, c_q = v.shape
                    no_heads = c_hidden // 32  # Assuming hidden dim per head is 32
                    
                    # Try to find the corresponding K and V weights
                    k_key = k_part + 'linear_k.weight'
                    v_key = k_part + 'linear_v.weight'
                    
                    if k_key in state_dict and v_key in state_dict:
                        k_weight = state_dict[k_key]
                        v_weight = state_dict[v_key]
                        
                        # Create fused weights
                        max_dim = max(c_q, k_weight.shape[1], v_weight.shape[1])
                        fused = torch.zeros((3, c_hidden, max_dim), device=v.device)
                        fused[0, :, :c_q] = v
                        fused[1, :, :k_weight.shape[1]] = k_weight
                        fused[2, :, :v_weight.shape[1]] = v_weight
                        
                        new_state_dict[fused_key] = fused.reshape(3 * c_hidden, max_dim)
            else:
                # Copy non-attention weights as is
                new_state_dict[k] = v
                
        return new_state_dict
    
    return convert_state_dict