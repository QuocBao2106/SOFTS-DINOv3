import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import kiến trúc mô hình đã sửa đổi của bạn
from models.SOFTS import Model

# Cấu hình giao diện trang web
st.set_page_config(
    page_title="Crypto SOFTS-DINOv3 Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Hệ thống Dự báo Chuỗi thời gian Crypto với SOFTS & DINOv3 Backbone")
st.write("Ứng dụng web trực tuyến tích hợp mô hình học sâu đã huấn luyện để dự báo biến động giá thị trường tiền mã hóa.")

# ================= 1. LOAD MÔ HÌNH AI ĐÃ XUẤT =================
class Configs:
    def __init__(self):
        self.seq_len = 96
        self.pred_len = 96
        self.d_model = 256
        self.d_core = 64
        self.dropout = 0.0
        self.use_norm = True

@st.cache_resource
def load_trained_model():
    configs = Configs()
    model = Model(configs)
    
    # Tìm file checkpoint.pth trong thư mục checkpoints/
    checkpoint_dir = "./checkpoints"
    checkpoint_path = None
    if os.path.exists(checkpoint_dir):
        for root, dirs, files in os.walk(checkpoint_dir):
            if "checkpoint.pth" in files:
                checkpoint_path = os.path.join(root, "checkpoint.pth")
                break
                
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
            model.eval()
            return model, f"Đã tải thành công mô hình từ: {checkpoint_path}"
        except Exception as e:
            return model, f"Lỗi khi load weights: {e}"
    else:
        return model, "Không tìm thấy file checkpoint.pth! Đang dùng trọng số ngẫu nhiên để demo."

model, load_status = load_trained_model()
st.sidebar.info(load_status)

# ================= 2. KHU VỰC TẢI DỮ LIỆU ĐẦU VÀO =================
st.subheader("1. Tải lên dữ liệu chuỗi thời gian (.csv)")
uploaded_file = st.file_uploader("Chọn file CSV dữ liệu các đồng coin (ví dụ: crypto_softs.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("🔍 **Xem trước 5 dòng dữ liệu đầu tiên:**")
    st.dataframe(df.head())
    
    # Chọn cột mục tiêu (Target column) để dự báo
    columns = [col for col in df.columns if col != 'date']
    target_col = st.selectbox("Chọn đồng coin/kênh cần dự báo trực quan:", columns, index=len(columns)-1)

    # ================= 3. NÚT CHẠY DỰ BÁO =================
    if st.button("🚀 Chạy mô hình dự báo tương lai"):
        with st.spinner("Mô hình đang xử lý qua DINOv3 Backbone và tính toán dự báo..."):
            try:
                # Lấy dữ liệu 96 bước thời gian gần nhất từ file CSV để làm đầu vào (Lookback window L=96)
                data_values = df[columns].values[-96:] # shape: (96, C)
                x_enc = torch.tensor(data_values, dtype=torch.float32).unsqueeze(0) # shape: (1, 96, C)
                
                # Chạy qua mô hình SOFTS
                with torch.no_grad():
                    # Các tham số giả lập mark cho hàm forecast
                    pred_out = model.forecast(x_enc, None, None, None) # shape: (1, pred_len, C)
                    
                # Chuyển kết quả về numpy để vẽ biểu đồ
                pred_numpy = pred_out.squeeze(0).numpy() # shape: (pred_len, C)
                target_idx = columns.index(target_col)
                
                history_vals = df[target_col].values[-96:]
                forecast_vals = pred_numpy[:, target_idx]
                
                st.success("✅ Dự báo hoàn tất!")
                
                # ================= 4. HIỂN THỊ KẾT QUẢ & BIỂU ĐỒ =================
                st.subheader(f"📊 Kết quả Dự báo cho mã: {target_col}")
                
                fig, ax = plt.subplots(figsize=(12, 5))
                time_history = range(len(history_vals))
                time_forecast = range(len(history_vals), len(history_vals) + len(forecast_vals))
                
                ax.plot(time_history, history_vals, label="Dữ liệu lịch sử (Ground Truth)", color="#1f77b4", linewidth=2)
                ax.plot(time_forecast, forecast_vals, label="Dự báo tương lai (Prediction - SOFTS+DINOv3)", color="#d62728", linewidth=2, linestyle="--")
                
                ax.set_title(f"Biểu đồ Chuỗi thời gian và Tầm nhìn dự báo H=96 cho {target_col}")
                ax.set_xlabel("Bước thời gian (Time steps)")
                ax.set_ylabel("Giá trị chuẩn hóa / Giá đóng cửa")
                ax.legend()
                ax.grid(True, linestyle=":", alpha=0.6)
                
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình dự báo: {e}")
else:
    st.info("💡 Vui lòng tải lên file dữ liệu `.csv` (ví dụ file `crypto_softs.csv` trong thư mục dataset của bạn) để bắt đầu thử nghiệm mô hình.")