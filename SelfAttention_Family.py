import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import math

from layers.Embed import TokenEmbedding, PatchEmbedding, DataEmbedding, PositionalEmbedding


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)





class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None): # xjt: x: bs, ts_d, seq_num, d_model 
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]  
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )   # xjt: time_in : bs, ts_d, seg_num, d_model ///// attn: bs * ts_d, seq_num, seq_num
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)  # xjt: dim_buffer = bs * seg_num, factor, d_model
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)   # xjt:  dim_send= bs * seg_num, ts_d,  d_model
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)   # xjt: bs * seg_num, ts_d, d_model

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
    
    

# xjt: TempAttention包含对inter-patch 和 intra-patch的attention计算
class TempAttention(nn.Module):
    def __init__(self, configs, win_size, patch_size, mask_flag=True, scale = None, attention_dropout = 0.05, output_attention = False) -> None:
        super().__init__()
        self.configs = configs
        self.win_size = win_size
        self.patch_size = patch_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        
    def forward(self, q_inter, k_inter, v_inter, q_intra, k_intra, v_intra, attn_mask=None, tau=None, delta=None):
        
        # inter-attention
        B, L, H, E = q_inter.shape  #   bs*n_vars, patch_num, n_head, d_model/n_head
        scale_inter = self.scale or 1. / sqrt(E)        
        score = torch.einsum("blhe,bshe->bhls", q_inter, k_inter) * scale_inter   # bs*n_vars, n_heads, patch_num, patch_num
        att_inter = self.dropout(torch.softmax(score, dim=-1))  # bs*n_vars, n_heads, patch_num, patch_num
        V_inter = torch.einsum("bhls,bshd->blhd", att_inter, v_inter)  # bs*n_vars, patch_num, n_heads, d_model/n_head
        
        # intra-attention
        B, L, H, E = q_intra.shape  #  bs*n_vars, patch_size, n_head, d_model/n_head
        scale_intra = self.scale or 1. / sqrt(E)
        score = torch.einsum("blhe,bshe->bhls", q_intra, k_intra) * scale_intra
        att_intra = self.dropout(torch.softmax(score, dim=-1))  # bs*n_vars, n_heads, patch_size, patch_size
        V_intra = torch.einsum("bhls,bshd->blhd", att_intra, v_intra)   # bs*n_vars, patch_size, n_heads, d_model/n_head
        
        
        # inter和intra的结果有不同的维度，需要进行upsample
        V_inter_upsam = repeat(V_inter, 'b patch_num n_heads d -> b (patch_num patch_size) n_heads d', patch_size=self.patch_size)
        V_intra_upsam = repeat(V_intra, 'b patch_size n_heads d -> b (patch_size patch_num) n_heads d', patch_num=self.win_size / self.patch_size)

        att_output = V_inter_upsam + V_intra_upsam  # bs*n_vars, patch_num*patch_size, n_heads, d_model/n_head
        
        return att_output, None        
    
    

class AttentionLayer_Xjt(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.n_heads = n_heads 
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        
        self.out_projection = nn.Linear(d_values * n_heads, d_model)      
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x_inter, x_intra, attn_mask):
        
        # inter-patch
        B, L, M = x_inter.shape
        H = self.n_heads
        q_inter, k_inter, v_inter = x_inter, x_inter, x_inter
        q_inter = self.query_projection(q_inter).view(B, L, H, -1) 
        k_inter = self.key_projection(k_inter).view(B, L, H, -1)
        v_inter = self.value_projection(v_inter).view(B, L, H, -1)  

        # intra-patch
        B, L, M = x_intra.shape
        q_intra, k_intra, v_intra = x_intra, x_intra, x_intra
        q_intra = self.query_projection(q_intra).view(B, L, H, -1) 
        k_intra = self.key_projection(k_intra).view(B, L, H, -1)
        v_intra = self.value_projection(v_intra).view(B, L, H, -1)
        
           
        out, att = self.inner_attention(
            q_inter,
            k_inter,
            v_inter,
            q_intra,
            k_intra,
            v_intra,
            attn_mask,
            tau=None,
            delta=None
        )
        # out: bs*n_vars, patch_num*patch_size, n_heads, d_model/n_head
        out = out.view(B, L, -1)  # bs*n_vars, patch_num*patch_size, d_model
        out = self.out_projection(out)
        return out, att


class TempAttention1(nn.Module):
    def __init__(self, configs, win_size, patch_size, inter_flag = True, mask_flag=True, scale = None, attention_dropout = 0.05, output_attention = False) -> None:
        super().__init__()
        self.configs = configs
        self.win_size = win_size
        self.patch_size = patch_size
        self.mask_flag = mask_flag
        self.inter_flag = inter_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        
    def forward(self, q, k, v, inter_flag = True, attn_mask=None, tau=None, delta=None):
        
        # inter-attention   or  intra-attention
        B, L, H, E = q.shape  #   bs*n_vars, patch_num, n_head, d_model/n_head
        scale_inter = self.scale or 1. / sqrt(E)        
        score = torch.einsum("blhe,bshe->bhls", q, k) * scale_inter   # bs*n_vars, n_heads, patch_num, patch_num
        att = self.dropout(torch.softmax(score, dim=-1))  # bs*n_vars, n_heads, patch_num, patch_num
        out = torch.einsum("bhls,bshd->blhd", att, v)  # bs*n_vars, patch_num, n_heads, d_model/n_head
        # if self.inter_flag:
        #     out = repeat(out, 'b patch_num n_heads d -> b (patch_num patch_size) n_heads d', patch_size=self.patch_size)
        # else:
        #     out = repeat(out, 'b patch_size n_heads d -> b (patch_size patch_num) n_heads d', patch_num=self.win_size // self.patch_size)
        
        return out, None        
    
    

class AttentionLayer_Xjt1(nn.Module):
    def __init__(self, attention, d_model, n_heads, win_size, patch_size = 16, d_keys=None, d_values=None):
        super(AttentionLayer_Xjt1, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.n_heads = n_heads 
        
        # xjt: add----------------------
        self.patch_size = patch_size
        self.win_size = win_size
        self.inter_value_embedding = TokenEmbedding(patch_size, d_model)
        self.intra_value_embedding = TokenEmbedding(self.win_size // patch_size, d_model)
        # ------------------------------
        
        
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        
        self.out_projection = nn.Linear(d_values * n_heads, d_model)      
        

    def forward(self, x_enc, attn_mask = None, tau=None, delta=None):
        # x_enc : bs, l, d_model 
        
        # inter-patch  or intra-patch
        B, L, M = x_enc.shape     # bs*n_vars, patch_num, d_model 
        H = self.n_heads     
        q_inter, k_inter, v_inter = x_enc, x_enc, x_enc
        q_inter = self.query_projection(q_inter).view(B, L, H, -1) 
        k_inter = self.key_projection(k_inter).reshape(B, L, H, -1)
        v_inter = self.value_projection(v_inter).reshape(B, L, H, -1)  
      
        out, att = self.inner_attention(
            q_inter,
            k_inter,
            v_inter,
            attn_mask,
            tau=None,
            delta=None
        )
        # out: bs*n_vars, patch_num*patch_size, n_heads, d_model/n_head
        new_L = out.shape[1]
        out = out.reshape(B, new_L, -1)  # bs*n_vars, patch_num*patch_size, d_model
        out = self.out_projection(out)
        return out, att
    

class TempAttention2(nn.Module):
    def __init__(self, configs, win_size, patch_size, attention_type='time', mask_flag=True, scale=None, dropout=0.1):
        super().__init__()
        self.attention_type = attention_type  # 'time'（inter/intra）或 'feature'
        self.win_size = win_size
        self.patch_size = patch_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        
        # 特征注意力专用参数
        if self.attention_type == 'feature':
            self.d_model = configs.d_model
            self.n_heads = configs.n_heads
            self.head_dim = self.d_model // self.n_heads

    def forward(self, q, k, v, attn_mask=None):
        if self.attention_type == 'time':
            # 原有时间维度注意力逻辑（inter/intra）
            B, L, H, E = q.shape  #   bs*n_vars, patch_num, n_head, d_model/n_head
            scale = self.scale or 1.0 / math.sqrt(E)
            score = torch.einsum("blhe,bshe->bhls", q, k) * scale
            if self.mask_flag and attn_mask is not None:
                score = score.masked_fill(attn_mask, -float('inf'))
            att = self.dropout(torch.softmax(score, dim=-1))
            out = torch.einsum("bhls,bshd->blhd", att, v) # bs*n_vars, patch_num, n_head, d_model/n_head
            return out, None
        
        elif self.attention_type == 'feature':
            # 新增特征维度注意力逻辑
            B, D, L, _ = q.shape  # b * patch_num,  n_var,  n_head, d_model/n_head
            scale = self.scale or 1.0 / math.sqrt(self.head_dim)
            score = torch.einsum("blhe,bshe->bhls", q, k) * scale
            att = self.dropout(torch.softmax(score, dim=-1))
            out = torch.einsum("bhls,bshd->blhd", att, v) # b * patch_num,  n_var,  n_head, d_model/n_head
            return out, None  
        
        
class AttentionLayer_Xjt2(nn.Module):
    def __init__(self, configs, attention, d_model, n_heads, win_size, patch_size=16):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.win_size = win_size
        self.patch_size = patch_size
        self.seq_len = configs.seq_len
        self.batch = configs.batch_size
        
        # 时间注意力（inter/intra）
        self.time_attention = attention(configs, win_size, patch_size, attention_type='time')
        
        # 特征注意力
        self.feature_attention = attention(configs, win_size, patch_size, attention_type='feature')
        
        # 投影层
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_time_intra, x_time_inter, attn_mask=None, tau=None, delta=None):
        # 时间维度注意力（inter/intra）
        # intra
        B, L, D = x_time_intra.shape  # b * n_var, patch_num, patch_size(patch_size -> d_model)
        q_time = self.query_proj(x_time_intra).view(B, L, self.n_heads, -1)
        k_time = self.key_proj(x_time_intra).view(B, L, self.n_heads, -1)
        v_time = self.value_proj(x_time_intra).view(B, L, self.n_heads, -1)
        time_intra_out, _ = self.time_attention(q_time, k_time, v_time, attn_mask)
        time_intra_out = time_intra_out.reshape(B, L, D)
        
        # inter
        B, L2, D = x_time_inter.shape   # b * n_var, patch_size, patch_num (patch_num -> d_model)
        q_time = self.query_proj(x_time_inter).view(B, L2, self.n_heads, -1)
        k_time = self.key_proj(x_time_inter).view(B, L2, self.n_heads, -1)
        v_time = self.value_proj(x_time_inter).view(B, L2, self.n_heads, -1)
        time_inter_out, _ = self.time_attention(q_time, k_time, v_time, attn_mask)
        time_inter_out = time_inter_out.reshape(B, L2, D)
        
        time_intra_out1 = repeat(time_intra_out, 'b patch_num d -> b (patch_num patch_size) d', patch_size=L2)
        # print(f'time_intra_out1 shape---origin:{x_time_intra.shape}, before:{time_intra_out.shape}, after:{time_intra_out1.shape}')
        
        time_inter_out1 = repeat(time_inter_out, 'b patch_size d -> b (patch_size patch_num) d', patch_num=L)
        # print(f'time_inter_out1 shape---origin:{x_time_inter.shape}, before:{time_inter_out.shape}, after:{time_inter_out1.shape}')
        
        time_out = time_intra_out1 + time_inter_out1  # b*n_var, patch_num*patch_size, d_model
        
        # print(f'time_out shape:{time_out.shape}')
        # 特征维度注意力 -- 时间维度依赖完成之后再来处理特征维度的依赖
        # x_feature = x_fea.permute(0, 2, 1)  # b * patch_num,  n_var,  patch_size (patch_size -> d_model)
        # q_feature = self.query_proj(x_feature).view(B, D, L, self.n_heads, -1)
        # k_feature = self.key_proj(x_feature).view(B, D, L, self.n_heads, -1)
        # v_feature = self.value_proj(x_feature).view(B, D, L, self.n_heads, -1)
        # feature_out, _ = self.feature_attention(q_feature, k_feature, v_feature)
        # feature_out = feature_out.reshape(B, L, D) # b * patch_num, n_var, d_model
        
        # time_out: b*n_var, patch_num*patch_size, d_model
        B_ori, D_ori, L_ori = time_out.shape
        x_feature = rearrange(time_out, '(b n_var) len d_model -> (b len) n_var d_model', b = self.batch) # b * len, n_vars, d_model
        B, D, L = x_feature.shape # 
        q_feature = self.query_proj(x_feature).view(B, D, self.n_heads, -1)
        k_feature = self.key_proj(x_feature).view(B, D, self.n_heads, -1)
        v_feature = self.value_proj(x_feature).view(B, D, self.n_heads, -1)
        feature_out, _ = self.feature_attention(q_feature, k_feature, v_feature) # b * len,  n_var,  n_head, d_model/n_head
        feature_out = feature_out.reshape(B_ori, D_ori, L_ori) # b * patch_num,  n_var, d_model
        
        
        # 合并结果
        out = feature_out
        out = self.out_proj(out)
        
        return time_intra_out, time_inter_out, out, None
    

##     2-21 主要改进是将embedding加入attionlayer中
class TempAttention3(nn.Module):
    def __init__(self, configs, patch_size, attention_type='time', mask_flag=True, scale=None, dropout=0.1):
        super().__init__()
        self.attention_type = attention_type  # 'time'（inter/intra）或 'feature'

        self.patch_size = patch_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        
        # 特征注意力专用参数
        if self.attention_type == 'feature':
            self.d_model = configs.d_model
            self.n_heads = configs.n_heads
            self.head_dim = self.d_model // self.n_heads

    def forward(self, q, k, v, attn_mask=None):
        if self.attention_type == 'time':
            # 原有时间维度注意力逻辑（inter/intra）
            B, L, H, E = q.shape  #   bs*n_vars, patch_num, n_head, d_model/n_head
            scale = self.scale or 1.0 / math.sqrt(E)
            score = torch.einsum("blhe,bshe->bhls", q, k) * scale
            if self.mask_flag and attn_mask is not None:
                score = score.masked_fill(attn_mask, -float('inf'))
            att = self.dropout(torch.softmax(score, dim=-1))
            out = torch.einsum("bhls,bshd->blhd", att, v) # bs*n_vars, patch_num, n_head, d_model/n_head
            return out, None
        
        elif self.attention_type == 'feature':
            # 新增特征维度注意力逻辑
            B, D, L, _ = q.shape  # b * patch_num,  n_var,  n_head, d_model/n_head
            scale = self.scale or 1.0 / math.sqrt(self.head_dim)
            score = torch.einsum("blhe,bshe->bhls", q, k) * scale
            att = self.dropout(torch.softmax(score, dim=-1))
            out = torch.einsum("bhls,bshd->blhd", att, v) # b * patch_num,  n_var,  n_head, d_model/n_head
            return out, None  
        
        
class AttentionLayer_Xjt3(nn.Module):
    def __init__(self, configs, attention, d_model, n_heads, patch_size=16, stride = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.patch_size = patch_size
        self.seq_len = configs.seq_len
        self.batch = configs.batch_size
        
        # 时间注意力（inter/intra）
        self.time_attention = attention(configs, patch_size, attention_type='time')
        
        ##  inter patch embedding + intra patch embedding
        padding = stride
        
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_size, stride, padding, configs.dropout)
        # inter patch embedding + intra patch embedding
        self.embedding_inter_patch = DataEmbedding(self.patch_size, configs.d_model, dropout=configs.dropout)
        self.embedding_intra_patch = DataEmbedding(self.seq_len // self.patch_size, configs.d_model, dropout=configs.dropout)
        
        self.embedding_all = DataEmbedding(self.d_model, self.d_model, dropout=configs.dropout)
         
        # 特征注意力
        self.feature_attention = attention(configs, patch_size, attention_type='feature')
        self.feature_mlp = ResBlock(configs=configs)
        
        # 投影层
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        
        # x : b, l, n_var, d_model
        x_time, x_fea = x, x
        batch, len, n_vars, d_model = x.shape
        # print(f'x_time shape:{x_time.shape}')
        # 时间维度注意力（inter/intra）
        # intra
        x_intra = x_time  # .permute(0, 2, 1) # b, n_var, len
        # x : b, l, n_var, d_model
        x_intra = rearrange(x_intra, 'b (n s) c d -> (b c) n s d', s = self.patch_size)  # b * n_var, patch_num, patch_size, d_model
        # x_intra = self.embedding_all(x_intra) # b * n_var, patch_num, patch_size, d_model
        
        B0, P0, S0, D0 = x_intra.shape # # b * n_var, patch_num, patch_size, d_model

        
        x_intra = x_intra.view(B0*P0, S0, D0) #  b * n_var * patch_num, patch_size, d_model
        x_intra = self.embedding_all(x_intra) 
        B, L, D = x_intra.shape

        
        q_time = self.query_proj(x_intra).view(B, L, self.n_heads, -1)
        k_time = self.key_proj(x_intra).view(B, L, self.n_heads, -1)
        v_time = self.value_proj(x_intra).view(B, L, self.n_heads, -1)
        time_intra_out, _ = self.time_attention(q_time, k_time, v_time, attn_mask)
        time_intra_out = time_intra_out.reshape(B, L, D) # b * n_var * patch_num, patch_size, d_model
        time_intra_out = time_intra_out.view(B0, P0, S0, D0)

        
        # inter
        x_inter = x_time # .permute(0, 2, 1)  # b, n_var, len
        x_inter = rearrange(x_inter, 'b (n s) c d -> (b c) s n d', s = self.patch_size)  # b * n_var, patch_size, patch_num, d_model
        # x_inter = self.embedding_all(x_inter)  # b * n_var, patch_size, patch_num, d_model
        
        B0, P0, S0, D0 = x_inter.shape # # b * n_var, patch_size, patch_num, d_model
        # print(f'x_inter shape:{x_inter.shape}')
        # x_inter = x_inter.view(B0*P0, S0, D0) #  b * n_var * patch_size, patch_num, d_model
        x_inter = x_inter.reshape(B0*P0, S0, D0) #  b * n_var * patch_size, patch_num, d_model
        
        x_inter = self.embedding_all(x_inter)
        B2, L2, D2 = x_inter.shape   # b * n_var * patch_size, patch_num, d_model
        
        q_time = self.query_proj(x_inter).view(B2, L2, self.n_heads, -1)
        k_time = self.key_proj(x_inter).view(B2, L2, self.n_heads, -1)
        v_time = self.value_proj(x_inter).view(B2, L2, self.n_heads, -1)
        time_inter_out, _ = self.time_attention(q_time, k_time, v_time, attn_mask)
        time_inter_out = time_inter_out.reshape(B2, L2, D2) # b * n_var * patch_size, patch_num, d_model
        time_inter_out = time_inter_out.view(B0, P0, S0, D0)
        
        time_intra_out1 = rearrange(time_intra_out, '(b c) n s d -> b (n s) c d', c = n_vars)
        time_inter_out1 = rearrange(time_inter_out, '(b c) s n d -> b (n s) c d', c = n_vars)
        
    
        time_out = time_intra_out1 + time_inter_out1  # b, patch_num*patch_size, n_var, d_model
        
        
        #  feature dependence -- attention
        # B1, P1, S1, D1 = time_out.shape  # b, patch_num*patch_size, n_var, d_model
        # x_feature = time_out
        # x_feature = rearrange(x_feature, '(b c) n s d -> (b n s) c d', b = self.batch) # b * patch_num * patch_size, n_vars, d_model
        # B, D, L = x_feature.shape # 
        # q_feature = self.query_proj(x_feature).view(B, D, self.n_heads, -1)
        # k_feature = self.key_proj(x_feature).view(B, D, self.n_heads, -1)
        # v_feature = self.value_proj(x_feature).view(B, D, self.n_heads, -1)
        # feature_out, _ = self.feature_attention(q_feature, k_feature, v_feature) # b * len,  n_var,  n_head, d_model/n_head
        # feature_out = feature_out.reshape(B, D, L) # b * patch_num * patch_size, n_vars, d_model
        # feature_out = feature_out.view(B1, P1, S1, D1)
        
        # fea dependence -- mlp
        B1, P1, S1, D1 = time_out.shape  # b, patch_num*patch_size, n_var, d_model
        x_feature = time_out
        feature_out = self.feature_mlp(x_feature.transpose(-2, -1)).transpose(-2, -1)
        
        
        # 合并结果
        out = feature_out
        out = self.out_proj(out)
        
        return  out, None  
    

class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        # x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x
