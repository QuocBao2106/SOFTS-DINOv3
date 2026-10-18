# SOFTS with DINOv3 Backbone: Thay thế STAR Encoder bằng DINOv3 cho Dự báo Chuỗi thời gian Đa biến

Dự án này thực hiện cải tiến kiến trúc mô hình từ bài báo **[SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion (NeurIPS 2024)](https://github.com/Secilia-Cxy/SOFTS)** bằng cách thay thế bộ mã hóa (Encoder/Backbone) nguyên bản bằng mô hình thị giác nền tảng **DINOv3** (`vit_small_patch16_dinov3.lvd1689m`).

---

## 🔍 Điểm cốt lõi: Chúng tôi đã thay thế những gì? (Architecture Comparison)

Theo **Algorithm 1** trong bài báo gốc, mô hình SOFTS bao gồm 4 giai đoạn:
1. **Chuẩn hóa thuận nghịch (RevIN):** Chuẩn hóa chuỗi đầu vào $X \in \mathbb{R}^{L \times C}$ về trung bình bằng 0, phương sai bằng 1 và đảo chiều ma trận thành $X \in \mathbb{R}^{C \times L}$[cite: 2].
2. **Series Embedding:** Chiếu tuyến tính từng kênh thời gian sang chiều ẩn: $S_0 = \text{Linear}(X) \in \mathbb{R}^{C \times d}$[cite: 2].
3. **Encoder / Backbone (Module STAR gốc):** Lặp qua các tầng Encoder, dùng `MLP` + `Stochastic Pooling` để gom các kênh thành biểu diễn cốt lõi toàn cục $o_i \in \mathbb{R}^{d'}$, sau đó nối (`Repeat_Concat`) và trộn (`MLP`) lại với từng chuỗi[cite: 2].
4. **Linear Predictor:** Dự báo chuỗi tương lai $\hat{Y} = \text{Linear}(S_N) \in \mathbb{R}^{C \times H}$ và hoàn nguyên chuẩn hóa RevIN[cite: 2].

### 🔄 Sự thay đổi trong dự án này (Giải quyết hạn chế ở Appendix F của bài báo)
Trong **Appendix F (Limitations and Future Works)** của bài báo gốc, nhóm tác giả chỉ ra rằng hiệu quả của SOFTS phụ thuộc chặt chẽ vào chất lượng của *Global Core Representation* và đề xuất hướng nghiên cứu thay thế chiến lược *Aggregate-Redistribute* của STAR[cite: 2]. 

Chúng tôi đã can thiệp trực tiếp vào file **`models/SOFTS.py`** để thay thế **Module STAR Encoder** bằng **`DINOv3BackboneEncoder`**:

| Thành phần kiến trúc | SOFTS Nguyên bản (`Secilia-Cxy/SOFTS`)[cite: 2] | SOFTS + DINOv3 (Bản cập nhật của chúng tôi) |
| :--- | :--- | :--- |
| **Input & Normalization** | RevIN + Transpose $(B, C, L)$[cite: 2] | **Giữ nguyên** RevIN + Transpose $(B, C, L)$[cite: 2] |
| **Series Embedding ($S_0$)** | `nn.Linear(seq_len, d_model)`[cite: 2] | **Giữ nguyên** `nn.Linear(seq_len, d_model)`[cite: 2] |
| **Backbone / Encoder** | Các tầng **`STAR`** (`MLP` + `Stoch_Pool`)[cite: 2] | **Thay thế hoàn toàn bằng `DINOv3BackboneEncoder`** (`vit_small_patch16_dinov3.lvd1689m`) |
| **Cơ chế trích xuất Core** | Gom cụm tuyến tính qua `Stochastic Pooling` trên chiều kênh[cite: 2] | Chuyển ma trận $(B, 1, C, L)$ thành ảnh 2D $(B, 3, 64, 64)$ $\rightarrow$ trích xuất đặc trưng không gian-thời gian qua **DINOv3 ViT** |
| **Output Predictor** | `nn.Linear(d_model, pred_len)`[cite: 2] | **Giữ nguyên** `nn.Linear(d_model, pred_len)`[cite: 2] |

---

## 🛠️ Chi tiết Code đã Cập nhật (File-by-File Code Changes)

### 1. File `models/SOFTS.py` (Trọng tâm cập nhật)
* **Loại bỏ:** Toàn bộ class `STAR` và kiến trúc `Encoder` / `EncoderLayer` dựa trên MLP nguyên bản[cite: 2].
* **Thêm mới class `DINOv3BackboneEncoder(nn.Module)`:**
  ```python
  # 1. Khởi tạo DINOv3 Pretrained Backbone từ thư viện timm
  self.dinov3 = timm.create_model(
      'vit_small_patch16_dinov3.lvd1689m',
      pretrained=True,
      num_classes=0,
      dynamic_img_size=True
  )
  # Đóng băng trọng số DINOv3 (Freeze Backbone) để tối ưu tài nguyên tính toán
  for param in self.dinov3.parameters():
      param.requires_grad = False

  # 2. Bộ chuyển đổi (Spatial & RGB Adapter) từ Chuỗi thời gian đa biến -> Ảnh 3 kênh
  self.to_rgb = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)

  # 3. Tầng chiếu và trộn đặc trưng (Core Projection & Series-Core Fusion MLP)
  self.dino_proj = nn.Linear(dino_embed_dim, d_model)
  self.fusion_mlp = nn.Sequential(
      nn.Linear(d_model * 2, d_model),
      nn.GELU(),
      nn.Linear(d_model, d_model)
  )

Cập nhật luồng xử lý trong hàm Model.forecast() (models/SOFTS.py):

Python
# Bước 1: Tạo Series Embedding ban đầu S_0 (Batch, Channels, d_model)
s_0 = self.dropout(self.series_embedding(x_transposed))

# Bước 2: Đưa dữ liệu qua DINOv3BackboneEncoder thay vì STAR Encoder
s_n = self.encoder(x_transposed, s_0)

# Bước 3: Dự báo qua Linear Predictor
dec_out = self.projection(s_n).permute(0, 2, 1)
2. File utils/tools.py (Bản vá tương thích môi trường)Dòng 54 (EarlyStopping.__init__): Cập nhật self.val_loss_min = np.Inf $\rightarrow$ self.val_loss_min = np.inf để tương thích với phiên bản NumPy >= 2.0.🚀 Hướng dẫn Cài đặt & Chạy Thực nghiệm
1. Cài đặt thư viện
Ngoài các thư viện cơ bản của SOFTS (torch, numpy, pandas, scikit-learn), cần cài đặt thêm timm để tải trọng số DINOv3:
pip install torch numpy pandas scikit-learn tqdm matplotlib einops patool timm transformers
2. Lệnh huấn luyện mô hình SOFTS + DINOv3
python run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path <ten_file_dataset.csv> \
  --model_id SOFTS_DINOv3_96_96 \
  --model SOFTS \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 1 \
  --enc_in 150 \
  --dec_in 150 \
  --c_out 150 \
  --d_model 256 \
  --d_core 64 \
  --d_ff 256 \
  --batch_size 32 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 10 \
  --patience 3 \
  --num_workers 0
📚 Tài liệu tham khảo (References)
SOFTS Original Paper: Lu Han, Xu-Yang Chen, Han-Jia Ye, De-Chuan Zhan. "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion." NeurIPS 2024[cite: 2].

Original Repository: https://github.com/Secilia-Cxy/SOFTS

[cite: 2]