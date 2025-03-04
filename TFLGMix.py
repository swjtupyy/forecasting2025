import torch
from torch import nn
from layers.Transformer_EncDec import Encoder_Xjt3, EncoderLayer_Xjt3,  scale_block
from layers.SelfAttention_Family import  AttentionLayer_Xjt3, TempAttention3
from layers.Embed import PatchEmbedding, DataEmbedding, PositionalEmbedding
from einops import rearrange, repeat
from layers.Autoformer_EncDec import series_decomp
from math import ceil


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
    

class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        
        self.configs = configs
        # Xjt: decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        padding = stride
        
        self.patch_size = 9 if configs.root_path == "dataset/illness/" else patch_len  # illness 的 seq_len=36
        
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        
        # inter patch embedding + intra patch embedding
        self.embedding_inter_patch = DataEmbedding(patch_len, configs.d_model, dropout=configs.dropout)
        self.embedding_intra_patch = DataEmbedding(self.seq_len // patch_len, configs.d_model, dropout=configs.dropout)
        
        self.embedding_position = PositionalEmbedding(configs.d_model)
        self.first_projection = nn.Linear(1, configs.d_model)
        
        # 将 b, l, n_var  --> b, l, n_var, 1 --> b, l, n_var, d_model
        self.start_fc = nn.Linear(in_features=1, out_features=configs.d_model)
        
         # 计算多尺度参数
        self.seg_num = ceil(configs.seq_len / self.patch_size)
        self.win_sizes = [2 ** l for l in range(configs.e_layers)]  # 指数增长窗口1, 2, 4
        
        # win_size 重新定义一下，其实这个叫dow_sampling更合适
        self.win_sizes = [1] + [2] * (configs.e_layers - 1)  # 首层1，后续层2
        
        
         # 构建多尺度编码器（核心修改）
        self.encoder1 = Encoder_Xjt3(
            [
                scale_block(
                    configs,
                    win_size=self.win_sizes[l],
                    d_model=configs.d_model,
                    n_heads=configs.n_heads,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout
                ) for l in range(configs.e_layers)
            ]
        )
        
        # 统一使用单个 Encoder
        # self.encoder = Encoder_Xjt3(
        #     [
        #         EncoderLayer_Xjt3(
        #             AttentionLayer_Xjt3(
        #                 configs,
        #                 TempAttention3,  # 传入类而非实例
        #                 configs.d_model,
        #                 configs.n_heads,
        #                 patch_size=self.patch_size
        #             ),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=nn.LayerNorm(configs.d_model)
        # )
        
        
        # Xjt: Linear -- for trend
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        # Prediction Head
        self.head_nf = configs.d_model * self.seq_len
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)
        # 预先创建多尺度预测头
        self.scale_heads = nn.ModuleList([
            FlattenHead(
                n_vars=configs.enc_in,
                nf=configs.d_model * (configs.seq_len // (2 ** l)),
                target_window=configs.pred_len,
                head_dropout=configs.dropout
            ).to(configs.device)  # 确保预测头在指定设备
            for l in range(configs.e_layers)
        ])
    

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Xjt: Decomp 
        seasonal, trend = self.decomp(x_enc)
        
        
        # seasonal -- predict
        seasonal = self.start_fc(seasonal.unsqueeze(-1))  # seasonal: bs, l, n_var, d_model
        
        # seasonal = seasonal.permute(0, 2, 1, 3)  # b, n_var, len, d_model
        enc_out, _ = self.encoder1(seasonal) # enc_out : b, patch_num*patch_size, n_var, d_model        //  b * n_vars, len, d_model  
        

        final_enc_out = torch.zeros(B, C, self.pred_len, 
                          device=x_enc.device)  # 正确初始化
        # print(f'final_enc_out: {final_enc_out.shape}')
        for i, (scale_out, head) in enumerate(zip(enc_out, self.scale_heads)):
            # 对每个尺度特征进行独立处理
            print(f'win_size:{self.win_sizes[i]}, configs.seq_len // (2 ** l):{self.seq_len // (2 ** i)}')
            print(f'scale_out: {scale_out.shape}')  # b, patch_num*patch_size, n_var, d_model
            
            # 步骤1：调整维度顺序
            scale_processed = rearrange(
                scale_out, 
                'b (n s) c d -> (b c) (n s) d',  # 按变量维度展开
                s=self.win_sizes[i]  # 动态计算当前尺度的片段长度
            )
            
            # 步骤2：特征重塑
            scale_processed = torch.reshape(
                scale_processed,
                (-1, self.n_vars, 
                scale_processed.shape[-2],  # 当前尺度的片段数 
                scale_processed.shape[-1])  # 特征维度
            )
            
            # 步骤3：维度置换适配预测头
            scale_processed = scale_processed.permute(0, 1, 3, 2)  # [bs, nvars, d_model, (patch_num * patch_size)]
            print(f'scale_processed: {scale_processed.shape}')
            final_enc_out += head(scale_processed)
            print(f'final_enc_out: {final_enc_out.shape}')
            # processed_features.append(scale_processed)
        
        
        # enc_out = rearrange(enc_out, 'b (n s) c d -> (b c) (n s) d', s = self.patch_size)   #  b * n_vars, len, d_model
        
        # enc_out = torch.reshape(
        #     enc_out, (-1, self.n_vars, enc_out.shape[-2], enc_out.shape[-1]))   # # z: [bs x nvars x (patch_num * patch_size) x d_model]
        # # enc_out: torch.Size([32, 8, 64, 48])
        # enc_out = enc_out.permute(0, 1, 3, 2)   # z: [bs, nvars, d_model, (patch_num * patch_size)]
        
        # print(f'enc_out: {enc_out.shape}')
        # Decoder
        
        dec_out = final_enc_out
        
        # trend -- predic
        trend = trend.permute(0, 2, 1)
        trend_out = self.Linear_Trend(trend)
        
        # Xjt: trend + seasonal
        dec_out += trend_out
        
        
        dec_out = dec_out.permute(0, 2, 1)
         # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out[:, -self.pred_len:, :]
