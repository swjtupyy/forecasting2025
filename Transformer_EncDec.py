import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import AttentionLayer_Xjt3, TempAttention3


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns




class EncoderLayer_Xjt(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_Xjt, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder_Xjt(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder_Xjt, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
    
    
class EncoderLayer_Xjt2(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_Xjt2, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_intra, x_inter, attn_mask=None, tau=None, delta=None):
        time_intra, time_inter, new_x, attn = self.attention(
            x_intra, x_inter,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = new_x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        
        return time_intra, time_inter, self.norm2(x + y), attn


class Encoder_Xjt2(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder_Xjt2, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_intra, x_inter, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        
        for attn_layer in self.attn_layers:
            x_intra, x_inter, x, attn = attn_layer(x_intra, x_inter, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

# Xjt：-----------------------------------------------------------------------------------------------------



class EncoderLayer_Xjt3(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer_Xjt3, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # 注意力层
        new_x, attn = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # 调整维度为 3D 输入
        batch, seq_len, n_vars, d_model = x.shape
        y = x.permute(0, 2, 1, 3)  # [batch, n_vars, seq_len, d_model]
        y = y.reshape(-1, seq_len, d_model)  # [batch * n_vars, seq_len, d_model]
        y = y.transpose(-1, -2)  # [batch * n_vars, d_model, seq_len]

        # 应用卷积层
        y = self.dropout(self.activation(self.conv1(y)))  # [batch * n_vars, d_ff, seq_len]
        y = self.dropout(self.conv2(y))  # [batch * n_vars, d_model, seq_len]

        # 恢复原始维度
        y = y.transpose(-1, -2)  # [batch * n_vars, seq_len, d_model]
        y = y.reshape(batch, n_vars, seq_len, d_model)
        y = y.permute(0, 2, 1, 3)  # [batch, seq_len, n_vars, d_model]

        # 残差连接和归一化
        return self.norm2(x + y), attn


class Encoder_Xjt31(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder_Xjt3, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


# 将时间序列分段按窗口合并，实现多尺度特征融合。
# eg: 输入8个段，窗口大小3 → 填充1个段 → 重组为3个窗口（每个含3段）→ 输出3个合并段。
class scale_block(nn.Module):
    def __init__(self, configs, win_size, d_model, n_heads, d_ff, dropout, seg_num=1, factor=0.5):
        super().__init__()
        self.win_size = win_size
        
        # 多尺度合并层
        if win_size > 1:
            self.merging = SegMerging(d_model, win_size)
        else:
            self.merging = None
        
        # 保持原有注意力结构
        self.attention = AttentionLayer_Xjt3(
            configs,
            TempAttention3,
            d_model,
            n_heads,
            patch_size=win_size
        )
        
        # 前馈网络
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 多尺度合并
        if self.merging is not None:
            x = self.merging(x)
        
        # 原有注意力处理
        x, _ = self.attention(x)
        return x



class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.win_size = win_size
        self.linear = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        # x shape: [B, D, L, C]
        b, l, d, c = x.shape # bs, l, n_var, d_model
        x = x.permute(0, 2, 1, 3).contiguous()  # 
        B, D, L, C = x.shape # # bs, n_var, l, d_model
        
        # 填充处理
        if L % self.win_size != 0:
            pad = self.win_size - (L % self.win_size)
            x = F.pad(x, (0,0,0,pad))
            L = L + pad
        
        # 窗口合并
        x = x.view(B, D, L//self.win_size, self.win_size, C)
        x = x.permute(0,1,3,2,4).contiguous()  # [B, D, win_size, L/win, C]
        x = x.view(B, D, -1, self.win_size*C)  # 合并特征维度
        x = self.norm(x)
        x = self.linear(x)
        
        return x.permute(0, 2, 1, 3)  # [B, D, L/win, C]

class Encoder_Xjt3(nn.Module):
    def __init__(self, attn_blocks):
        super().__init__()
        self.attn_blocks = nn.ModuleList(attn_blocks)

    def forward(self, x):
        encoder_x = []
        # encoder_x.append(x)
        for block in self.attn_blocks:  # 这个地方的问题：你后一个的值应该是在前一个的基础上进行的，所以e_layer如果为3，最后一个的长度就是48 // 4 = 12了，而第三层应该为24才对， 96 --> 96 / 2 = 48 --> 48 / 2 = 24()
            x = block(x)
            encoder_x.append(x)
        return encoder_x, None


# Xjt：-----------------------------------------------------------------------------------------------------





class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
