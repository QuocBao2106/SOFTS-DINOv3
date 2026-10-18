import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DINOv3BackboneEncoder(nn.Module):
    """
    Bộ mã hóa (Encoder/Backbone) sử dụng DINOv3 thay thế cho các lớp STAR.
    Biến đổi chuỗi thời gian đa biến (B, C, L) -> Ảnh 2D (B, 3, H, W) -> DINOv3 -> Đặc trưng (B, C, d_model).
    """
    def __init__(self, seq_len, d_model, img_size=64, freeze_backbone=True):
        super().__init__()
        self.img_size = img_size
        
        # 1. Tải mô hình DINOv3 chính thức từ thư viện timm (bản Small để chạy mượt trên CPU)
        print(">>> Đang tải DINOv3 Backbone (vit_small_patch16_dinov3.lvd1689m)...")
        self.dinov3 = timm.create_model(
            'vit_small_patch16_dinov3.lvd1689m',
            pretrained=True,
            num_classes=0,          # Bỏ lớp phân loại ảnh cuối cùng để lấy vector đặc trưng
            dynamic_img_size=True   # Cho phép nhận kích thước ảnh linh hoạt
        )
        dino_embed_dim = self.dinov3.num_features  # Với ViT-Small, chiều đặc trưng là 384

        # Đóng băng (Freeze) trọng số của DINOv3 để không phải huấn luyện lại 21 triệu tham số trên CPU
        if freeze_backbone:
            for param in self.dinov3.parameters():
                param.requires_grad = False

        # 2. Adapter chuyển đổi từ 1 kênh chuỗi thời gian sang 3 kênh (giả lập RGB cho DINOv3)
        self.to_rgb = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)

        # 3. Các lớp chiếu (Projection) kết hợp đặc trưng cục bộ của từng kênh và đặc trưng toàn cục từ DINOv3
        self.dino_proj = nn.Linear(dino_embed_dim, d_model)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_raw, s_0):
        """
        x_raw: Chuỗi thời gian đầu vào (Batch, Channels, Seq_len)
        s_0  : Series embedding ban đầu (Batch, Channels, d_model)
        """
        B, C, L = x_raw.shape

        # Bước A: Co giãn ma trận (B, 1, C, L) thành kích thước ảnh vuông (B, 1, img_size, img_size)
        x_2d = x_raw.unsqueeze(1)  # (B, 1, C, L)
        x_resized = F.interpolate(x_2d, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        # Bước B: Chuyển sang 3 kênh RGB (B, 3, img_size, img_size)
        x_rgb = self.to_rgb(x_resized)

        # Bước C: Đưa qua DINOv3 Backbone để trích xuất đặc trưng toàn cục
        dino_feat = self.dinov3(x_rgb)               # (B, dino_embed_dim = 384)
        global_core = self.dino_proj(dino_feat)      # (B, d_model)

        # Bước D: Phân bổ (Redistribute) đặc trưng DINOv3 về lại cho từng kênh C
        global_core_expanded = global_core.unsqueeze(1).expand(-1, C, -1)  # (B, C, d_model)

        # Bước E: Trộn (Fuse) đặc trưng từng chuỗi s_0 với đặc trưng từ DINOv3 + Residual connection
        fused = torch.cat([s_0, global_core_expanded], dim=-1)             # (B, C, 2 * d_model)
        out = self.norm(s_0 + self.fusion_mlp(fused))                      # (B, C, d_model)
        return out


class Model(nn.Module):
    """
    Mô hình SOFTS đã được thay thế Encoder/Backbone bằng DINOv3.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.use_norm = configs.use_norm

        # 1. Series Embedding (Giữ nguyên theo phương trình (1) của bài báo SOFTS)
        self.series_embedding = nn.Linear(self.seq_len, self.d_model)
        self.dropout = nn.Dropout(configs.dropout)

        # 2. Backbone / Encoder: Thay thế các lớp STAR bằng DINOv3BackboneEncoder
        self.encoder = DINOv3BackboneEncoder(
            seq_len=self.seq_len,
            d_model=self.d_model,
            img_size=64,
            freeze_backbone=True
        )

        # 3. Linear Predictor (Giữ nguyên theo dòng 8 Algorithm 1 của bài báo SOFTS)
        self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Áp dụng Reversible Instance Normalization (RevIN) theo Mục 3.1 của bài báo SOFTS
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Đảo chiều từ (Batch, Seq_len, Channels) sang (Batch, Channels, Seq_len)
        x_transposed = x_enc.permute(0, 2, 1)

        # Bước 1: Tạo Series Embedding ban đầu S_0
        s_0 = self.dropout(self.series_embedding(x_transposed))

        # Bước 2: Đưa qua DINOv3 Encoder/Backbone
        s_n = self.encoder(x_transposed, s_0)

        # Bước 3: Dự báo tương lai bằng Linear Predictor và đảo chiều lại về (Batch, Pred_len, Channels)
        dec_out = self.projection(s_n).permute(0, 2, 1)

        # Hoàn nguyên chuẩn hóa (De-normalization) của RevIN
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # (Batch, Pred_len, Channels)