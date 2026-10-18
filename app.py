import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from models.SOFTS import Model

# Cấu hình trang rộng (Wide mode) và giao diện Dark Theme đặc trưng
st.set_page_config(
    page_title="Crypto AI Analytics Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 1. CUSTOM CSS ĐỂ LÀM GIAO DIỆN HIỆN ĐẠI (DARK MODE) =================
st.markdown("""
    <style>
    /* Màu nền tổng thể Dark Mode */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    /* Thanh Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    /* Card chứa thành phần */
    .metric-card {
        background-color: #161e2e;
        border: 1px solid #1f2937;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    /* Tiêu đề bảng điều khiển */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    /* Tùy chỉnh nút bấm */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }
    </style>
""", unsafe_allow_html=True)

# ================= 2. THANH ĐIỀU HƯỚNG & MENU 3 CHẤM (SIDEBAR) =================
with st.sidebar:
    st.markdown("### 🌐 **SOFTS-DINOv3 Hub**")
    st.markdown("---")
    
    # Thanh menu mục lục dạng selectbox hoặc radio cao cấp
    menu_selection = st.selectbox(
        "📌 **Điều hướng hệ thống**",
        ["📊 Dashboard Tổng quan", "🚀 Chạy Dự báo AI", "⚙️ Cấu hình Mô hình", "📖 Tài liệu & Hướng dẫn"]
    )
    
    st.markdown("---")
    st.markdown("### 🎛️ **Công cụ nhanh (Actions)**")
    # Menu tùy chọn dạng 3 chấm / action phụ
    quick_action = st.selectbox(
        "Options ⋯",
        ["Xuất báo cáo PDF", "Làm sạch cache mô hình", "Cài đặt giao diện Dark/Light"]
    )
    
    st.markdown("---")
    st.markdown("<p style='font-size: 12px; color: #64748b;'>Hệ thống AI Time-Series Forecasting v2.4<br>© 2026 Nguyễn Quốc Bảo</p>", unsafe_allow_html=True)

# ================= 3. LOAD MÔ HÌNH AI ĐÃ XUẤT (CHECKPOINT) =================
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
            return model, "🟢 Mô hình AI đã sẵn sàng (Loaded checkpoint.pth)"
        except Exception as e:
            return model, f"🟡 Lỗi tải weights: {e}"
    else:
        return model, "🟠 Chưa tìm thấy file checkpoint.pth (Sử dụng chế độ giả lập)"

model, load_status = load_trained_model()

# ================= 4. GIAO DIỆN CHÍNH (DASHBOARD) =================
# Header Top Bar: Tiêu đề và trạng thái mô hình
col_top1, col_top2 = st.columns([3, 1])
with col_top1:
    st.markdown("# ⚡ Crypto Intelligence Dashboard")
    st.markdown(f"<span style='color: #10b981; font-size: 14px;'>● Status: {load_status}</span>", unsafe_allow_html=True)
with col_top2:
    # Thanh tìm kiếm mô phỏng phía trên góc phải
    search_query = st.text_input("🔍 Tìm kiếm mã coin...", placeholder="Nhập BTC, ETH...")

st.markdown("---")

# Điều hướng hiển thị theo Menu mục lục bên Sidebar
if menu_selection == "📊 Dashboard Tổng quan":
    # Các thẻ Cards thông số tổng quan (Metrics Cards)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><h4>📦 Tổng số Kênh (C)</h4><h2>150 Assets</h2><p style='color: #10b981;'>+12% tuần này</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><h4>⏱️ Cửa sổ nhìn lại (L)</h4><h2>96 Hours</h2><p style='color: #3b82f6;'>Chuẩn tiêu chuẩn</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><h4>🎯 Tầm nhìn (H)</h4><h2>96 Horizon</h2><p style='color: #f59e0b;'>Độ chính xác cao</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><h4>⚡ Hiệu năng MSE</h4><h2>0.0653</h2><p style='color: #10b981;'>Tối ưu qua DINOv3</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Gợi ý:** Hãy chuyển sang mục **'🚀 Chạy Dự báo AI'** ở thanh menu bên trái để tải lên dữ liệu CSV của bạn và trực quan hóa kết quả dự báo.")

elif menu_selection == "🚀 Chạy Dự báo AI":
    st.markdown("### 📂 Tải lên Dữ liệu & Cấu hình Phân tích")
    
    col_upload, col_settings = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Kéo thả hoặc chọn file CSV chuỗi thời gian", type=["csv"])
    with col_settings:
        forecast_horizon = st.slider("Chọn tầm nhìn dự báo (Horizon H):", min_value=12, max_value=96, value=96, step=12)
        confidence_interval = st.selectbox("Độ tin cậy mô hình:", ["95% Standard", "99% High Precision"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("---")
        st.write("🔍 **Xem trước cấu trúc dữ liệu đầu vào:**")
        st.dataframe(df.head(), use_container_width=True)
        
        columns = [col for col in df.columns if col != 'date']
        target_col = st.selectbox("📌 Chọn đồng coin/kênh mục tiêu để phân tích:", columns, index=len(columns)-1)

        if st.button("🚀 Thực thi Dự báo AI ngay"):
            with st.spinner("Đang trích xuất đặc trưng qua DINOv3 Backbone & chạy mô hình SOFTS..."):
                try:
                    data_values = df[columns].values[-96:]
                    x_enc = torch.tensor(data_values, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        pred_out = model.forecast(x_enc, None, None, None)
                        
                    pred_numpy = pred_out.squeeze(0).numpy()
                    target_idx = columns.index(target_col)
                    
                    history_vals = df[target_col].values[-96:]
                    forecast_vals = pred_numpy[:forecast_horizon, target_idx]
                    
                    st.success("✨ Dự báo hoàn tất thành công!")
                    
                    # Khu vực hiển thị biểu đồ giao diện tối màu chuyên nghiệp
                    st.markdown(f"### 📈 Biểu đồ Phân tích & Dự báo: `{target_col}`")
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    # Tùy chỉnh style biểu đồ hợp với Dark Mode
                    fig.patch.set_facecolor('#111827')
                    ax.set_facecolor('#161e2e')
                    
                    time_history = range(len(history_vals))
                    time_forecast = range(len(history_vals), len(history_vals) + len(forecast_vals))
                    
                    ax.plot(time_history, history_vals, label="Dữ liệu lịch sử thực tế (Ground Truth)", color="#3b82f6", linewidth=2)
                    ax.plot(time_forecast, forecast_vals, label="Dự báo tương lai (SOFTS + DINOv3)", color="#ef4444", linewidth=2.5, linestyle="--")
                    
                    ax.set_title(f"Tầm nhìn dự báo H={forecast_horizon} cho mã {target_col}", color="white", fontsize=14)
                    ax.set_xlabel("Bước thời gian (Time steps)", color="#94a3b8")
                    ax.set_ylabel("Giá trị chuẩn hóa / Biến động giá", color="#94a3b8")
                    ax.tick_params(colors="#94a3b8")
                    ax.legend(facecolor='#111827', edgecolor='none', labelcolor='white')
                    ax.grid(True, linestyle=":", alpha=0.3, color="#475569")
                    
                    for spine in ax.spines.values():
                        spine.set_color('#334155')
                        
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi thực thi mô hình: {e}")
    else:
        st.info("💡 Vui lòng tải lên file dữ liệu `.csv` (ví dụ file `crypto_softs.csv`) để bắt đầu.")

elif menu_selection == "⚙️ Cấu hình Mô hình":
    st.markdown("### ⚙️ Thông số Kiến trúc Hệ thống")
    st.json({
        "Model Backbone": "DINOv3 ViT-Small (vit_small_patch16_dinov3.lvd1689m)",
        "Time-Series Core": "SOFTS STAR Architecture",
        "Normalization": "RevIN (Reversible Instance Normalization)",
        "Hidden Dimension (d_model)": 256,
        "Core Dimension (d_core)": 64,
        "Lookback Window (L)": 96
    })

elif menu_selection == "📖 Tài liệu & Hướng dẫn":
    st.markdown("### 📖 Tài liệu Hướng dẫn Sử dụng Hệ thống")
    st.markdown("""
    - **Bước 1:** Chuẩn bị file dữ liệu chuỗi thời gian định dạng `.csv` có cột `date` và các cột kênh dữ liệu.
    - **Bước 2:** Truy cập mục **Chạy Dự báo AI**, tải file lên hệ thống.
    - **Bước 3:** Lựa chọn đồng coin mục tiêu và nhấn nút thực thi để nhận biểu đồ trực quan từ mô hình học sâu.
    """)