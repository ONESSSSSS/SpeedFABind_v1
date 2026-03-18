import torch
from torch import nn
from torch.nn import LayerNorm, Linear
import numpy as np
import triton
import triton.language as tl
import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple

# from models.model_utils import Transition, InteractionModule
from models.model_utils import permute_final_dims, Attention, Transition, InteractionModule
# p_attention_stream = torch.cuda.Stream()


class CrossAttentionModule(nn.Module):
    # CrossAttentionModule 类实现了一个跨注意力机制，结合了行注意力和三角注意力，用于处理节点嵌入和节点对嵌入。通过各种注意力块和过渡模块，动态计算节点之间的交互，从而增强模型的表达能力。
    def __init__(self, node_hidden_dim, pair_hidden_dim, rm_layernorm=False, keep_trig_attn=False, dist_hidden_dim=32, normalize_coord=None):
        super().__init__()
        self.pair_hidden_dim = pair_hidden_dim
        self.keep_trig_attn = keep_trig_attn

        if keep_trig_attn:
            self.triangle_block_row = RowTriangleAttentionBlock(pair_hidden_dim, dist_hidden_dim, rm_layernorm=rm_layernorm)
            self.triangle_block_column = RowTriangleAttentionBlock(pair_hidden_dim, dist_hidden_dim, rm_layernorm=rm_layernorm)

        self.p_attention_block = RowAttentionBlock(node_hidden_dim, pair_hidden_dim, rm_layernorm=rm_layernorm)
        self.c_attention_block = RowAttentionBlock(node_hidden_dim, pair_hidden_dim, rm_layernorm=rm_layernorm)
        self.p_transition = Transition(node_hidden_dim, 2, rm_layernorm=rm_layernorm)
        self.c_transition = Transition(node_hidden_dim, 2, rm_layernorm=rm_layernorm)
        self.pair_transition = Transition(pair_hidden_dim, 2, rm_layernorm=rm_layernorm)
        self.inter_layer = InteractionModule(node_hidden_dim, pair_hidden_dim, 32, opm=False, rm_layernorm=rm_layernorm)

    def forward(self, 
                p_embed_batched, p_mask,
                c_embed_batched, c_mask,
                pair_embed, pair_mask,
                c_c_dist_embed=None, p_p_dist_embed=None):
    
        with torch.profiler.record_function("triangle_block_row"):
         if self.keep_trig_attn:
            pair_embed = self.triangle_block_row(pair_embed=pair_embed,
                                                pair_mask=pair_mask,
                                                dist_embed=c_c_dist_embed)
            pair_embed = self.triangle_block_row(pair_embed=pair_embed.transpose(-2, -3),
                                                pair_mask=pair_mask.transpose(-1, -2),
                                                dist_embed=p_p_dist_embed).transpose(-2, -3)
        
        # with torch.profiler.record_function("pc_attention_block"):
            # p_attention_stream = torch.cuda.Stream()
            # c_attention_stream = torch.cuda.Stream()

            # p_attention_block
            # with torch.profiler.record_function("p_attention_block"):
            #     with torch.cuda.stream(p_attention_stream):
            #         p_embed_batched = self.p_attention_block(
            #             node_embed_i=p_embed_batched,
            #             node_embed_j=c_embed_batched,
            #             pair_embed=pair_embed,
            #             pair_mask=pair_mask,
            #             node_mask_i=p_mask
            #         )
            #     p_embed_batched = p_embed_batched + self.p_transition(p_embed_batched) 

            # c_attention_block
            # with torch.profiler.record_function("c_attention_block"):
            #     with torch.cuda.stream(c_attention_stream):
            #         c_embed_batched = self.c_attention_block(
            #             node_embed_i=c_embed_batched,
            #             node_embed_j=p_embed_batched,
            #             pair_embed=pair_embed.transpose(-2, -3),
            #             pair_mask=pair_mask.transpose(-1, -2),
            #             node_mask_i=c_mask
            #         )
            #     c_embed_batched = c_embed_batched + self.c_transition(c_embed_batched)

        with torch.profiler.record_function("p_attention_block"):
            p_embed_batched = self.p_attention_block(node_embed_i=p_embed_batched,
                                                node_embed_j=c_embed_batched,
                                                pair_embed=pair_embed,
                                                pair_mask=pair_mask,
                                                node_mask_i=p_mask)
        with torch.profiler.record_function("c_attention_block"):
            c_embed_batched = self.c_attention_block(node_embed_i=c_embed_batched,
                                                node_embed_j=p_embed_batched,
                                                pair_embed=pair_embed.transpose(-2, -3),
                                                pair_mask=pair_mask.transpose(-1, -2),
                                                node_mask_i=c_mask)
    
        with torch.profiler.record_function("pc_transition"):
            p_embed_batched = p_embed_batched + self.p_transition(p_embed_batched) 
            c_embed_batched = c_embed_batched + self.c_transition(c_embed_batched)

        with torch.profiler.record_function("inter_layer"):
            pair_embed = pair_embed + self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)[0]
        with torch.profiler.record_function("pair_transition"):
            pair_embed = self.pair_transition(pair_embed) * pair_mask.to(torch.float).unsqueeze(-1)
        return p_embed_batched, c_embed_batched, pair_embed


class RowTriangleAttentionBlock(nn.Module):
    inf = 1e9

    def __init__(self, pair_hidden_dim, dist_hidden_dim, attention_hidden_dim=32, no_heads=4, dropout=0.1, rm_layernorm=False):
        super(RowTriangleAttentionBlock, self).__init__()
        self.no_heads = no_heads
        self.attention_hidden_dim = attention_hidden_dim
        self.dist_hidden_dim = dist_hidden_dim
        self.pair_hidden_dim = pair_hidden_dim

        self.rm_layernorm = rm_layernorm
        if not self.rm_layernorm:
            self.layernorm = LayerNorm(pair_hidden_dim)

        self.linear = Linear(dist_hidden_dim, self.no_heads)
        self.linear_g = Linear(dist_hidden_dim, self.no_heads)
        self.dropout = nn.Dropout(dropout)
        self.mha = Attention(
            pair_hidden_dim, pair_hidden_dim, pair_hidden_dim, attention_hidden_dim, no_heads
        )

    def forward(self, pair_embed, pair_mask, dist_embed):
        if not self.rm_layernorm:
            pair_embed = self.layernorm(pair_embed)  # (*, I, J, C_pair)

        mask_bias = (self.inf * (pair_mask.to(torch.float) - 1))[..., :, None, None, :]  # (*, I, 1, 1, J)
        dist_bias = self.linear(dist_embed) * self.linear_g(dist_embed).sigmoid()  # (*, J, J, H)
        dist_bias = permute_final_dims(dist_bias, [2, 1, 0])[..., None, :, :, :]  # (*, 1, H, J, J)

        pair_embed = pair_embed + self.dropout(self.mha(
            q_x=pair_embed,  # [*, I, J, C_pair]
            kv_x=pair_embed,  # [*, I, J, C_pair]
            biases=[mask_bias, dist_bias]  # List of [*, I, H, J, J]
        )) * pair_mask.to(torch.float).unsqueeze(-1)  # (*, I, J, C_pair)

        return pair_embed


# 假设 LayerNorm、Linear、Attention 和 permute_final_dims 已在其他地方定义













































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

        self.linear = Linear(pair_hidden_dim, self.no_heads)
        self.linear_g = Linear(pair_hidden_dim, self.no_heads)

        self.dropout = nn.Dropout(dropout)

        self.mha = Attention(node_hidden_dim, node_hidden_dim, node_hidden_dim, attention_hidden_dim, no_heads)

    def forward(self, node_embed_i, node_embed_j, pair_embed, pair_mask, node_mask_i):
        if not self.rm_layernorm:
            node_embed_i = self.layernorm_node_i(node_embed_i)  # (*, I, C_node)
            node_embed_j = self.layernorm_node_j(node_embed_j)  # (*, J, C_node)
            pair_embed = self.layernorm_pair(pair_embed)  # (*, I, J, C_pair)

        with torch.profiler.record_function("maskbise"):
            # mask_bias = (self.inf * (pair_mask.to(torch.float) - 1))[..., None, :, :]  # (*, 1, I, J)
            # 将 pair_mask 转换为浮点数
            pair_mask_float = pair_mask.to(torch.float)

            # 减去 1
            adjusted_mask = pair_mask_float - 1

            # 乘以 self.inf
            mask_values = self.inf * adjusted_mask

            # 调整维度
            mask_bias = mask_values[..., None, :, :]
        with torch.profiler.record_function("pairbise"):
            # pair_bias = self.linear(pair_embed) * self.linear_g(pair_embed).sigmoid()  # (*, I, J, H)
            # pair_bias = permute_final_dims(pair_bias, [2, 0, 1])  # (*, H, I, J)
            
            # print("pair_embed shape:", pair_embed.shape)
            # linear_output = self.linear(pair_embed)
            # linear_g_output = self.linear_g(pair_embed)
            # print("linear weight shape:", self.linear.weight.shape)  # 打印linear权重的形状
            # print("linear_g weight shape:", self.linear_g.weight.shape)  # 打印linear_g权重的形状
            # sigmoid_output = linear_g_output.sigmoid()
            # pair_bias = linear_output * sigmoid_output  # (*, I, J, H)
            # pair_bias = permute_final_dims(pair_bias, [2, 0, 1])  # (*, H, I, J)


            #更改拼接
            # reshaped_pair_embed = pair_embed.reshape(-1, 128)
            combined_weight = torch.cat((self.linear.weight, self.linear_g.weight), dim=0)
            combined_output = torch.matmul(pair_embed, combined_weight.T)
            split_dim = combined_output.shape[-1] // 2
            left_part = combined_output[..., :split_dim]   # 左半部分
            right_part = combined_output[..., split_dim:]   # 右半部分
            # 4. 对右半部分应用sigmoid
            sigmoid_output = torch.sigmoid(right_part)

            # 5. 左半部分与经过sigmoid的右半部分进行逐元素相乘
            pair_bias = left_part * sigmoid_output
            pair_bias = permute_final_dims(pair_bias, [2, 0, 1])  # (*, H, I, J)

        # return pair_bias

        mha_output = self.mha(
            q_x=node_embed_i,
            kv_x=node_embed_j,
            biases=[mask_bias, pair_bias]
        )

        # 第二步：对多头注意力机制的输出应用Dropout
        dropout_output = self.dropout(mha_output)

        # 第三步：将node_mask_i转换为torch.float类型
        node_mask_i_float = node_mask_i.to(torch.float)

        # 第四步：在最后一维上增加一个维度（unsqueeze操作）
        if isinstance(node_mask_i_float, np.ndarray):
          node_mask_i_float= torch.from_numpy(node_mask_i_float)
        node_mask_i_unsqueezed = node_mask_i_float.unsqueeze(-1)

        # 第五步：将经过Dropout处理的多头注意力机制输出与处理后的node_mask_i相乘
        multiplied_output = dropout_output * node_mask_i_unsqueezed

        # 第六步：将原始的node_embed_i与相乘后的结果相加，得到最终的node_embed_i
        node_embed_i = node_embed_i + multiplied_output
        return node_embed_i


def check_convert(g):
        if isinstance(g, torch.Tensor) :
            return g
        else:
            if isinstance(g, np.ndarray):
                g = torch.from_numpy(g)
            
            return g