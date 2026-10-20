import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- CẤU HÌNH GIAO DIỆN TRANG WEB ---
st.set_page_config(
    page_title="Hệ thống Dự đoán Bệnh tim",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Hệ thống Hỗ trợ Chẩn đoán Bệnh Tim Mạch")
st.markdown("""
Ứng dụng sử dụng mô hình học máy **Random Forest** được huấn luyện trên bộ dữ liệu chuẩn **UCI Cleveland Heart Disease** (303 bệnh nhân) để dự đoán nguy cơ mắc bệnh tim dựa trên các chỉ số lâm sàng.
""")

# --- TẢI DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH TỰ ĐỘNG ---
@st.cache_resource
def load_and_train_model():
    # Tìm đường dẫn file data
    data_path = "data/heart_disease_uci.csv"
    if not os.path.exists(data_path):
        data_path = "../data/heart_disease_uci.csv" # Nếu chạy từ thư mục notebooks
        
    df = pd.read_csv(data_path)
    
    # Xử lý dữ liệu khuyết (giống như ở Ngày 3)
    for col in ['ca', 'thal']:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Huấn luyện mô hình Random Forest tối ưu
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf, X.columns

rf_model, feature_names = load_and_train_model()

st.divider()
st.subheader("📝 Nhập thông tin chỉ số lâm sàng của bệnh nhân:")

# --- TẠO CÁC THANH NHẬP LIỆU (INPUT CONTROLS) CHO 13 THUỘC TÍNH ---
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Tuổi (Age)", min_value=20, max_value=90, value=55)
    sex = st.selectbox("Giới tính (Sex)", options=[1, 0], format_func=lambda x: "Nam (1)" if x == 1 else "Nữ (0)")
    cp = st.selectbox("Loại đau ngực (Chest Pain Type - cp)", options=[1, 2, 3, 4], 
                      format_func=lambda x: {1: "1: Đau thắt ngực điển hình", 2: "2: Đau thắt ngực không điển hình", 3: "3: Đau không do tim", 4: "4: Không triệu chứng"}[x])
    trestbps = st.slider("Huyết áp tâm thu lúc nghỉ (mmHg)", min_value=90, max_value=200, value=130)
    chol = st.slider("Cholesterol huyết thanh (mg/dl)", min_value=120, max_value=600, value=240)
    fbs = st.selectbox("Đường huyết lúc đói > 120 mg/dl (fbs)", options=[1, 0], format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")
    restecg = st.selectbox("Điện tâm đồ lúc nghỉ (restecg)", options=[0, 1, 2], format_func=lambda x: {0: "0: Bình thường", 1: "1: Bất thường sóng ST-T", 2: "2: Phì đại thất trái"}[x])

with col2:
    thalach = st.slider("Nhịp tim tối đa đạt được (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Đau thắt ngực khi gắng sức (exang)", options=[1, 0], format_func=lambda x: "Có (1)" if x == 1 else "Không (0)")
    oldpeak = st.number_input("Độ chênh đoạn ST khi gắng sức (oldpeak)", min_value=0.0, max_value=6.5, value=1.0, step=0.1)
    slope = st.selectbox("Độ dốc đoạn ST (slope)", options=[1, 2, 3], format_func=lambda x: {1: "1: Dốc lên", 2: "2: Đi ngang", 3: "3: Dốc xuống"}[x])
    ca = st.selectbox("Số lượng mạch máu chính nhuốm màu (ca)", options=[0.0, 1.0, 2.0, 3.0])
    thal = st.selectbox("Tưới máu cơ tim (thal)", options=[3.0, 6.0, 7.0], format_func=lambda x: {3.0: "3: Bình thường", 6.0: "6: Khiếm khuyết cố định", 7.0: "7: Khiếm khuyết hồi phục"}[x])

st.divider()

# --- NÚT BẤM DỰ ĐOÁN ---
if st.button("🔍 Chẩn đoán Nguy cơ Bệnh Tim", type="primary", use_container_width=True):
    # Gom dữ liệu đầu vào thành DataFrame khớp với 13 cột của mô hình
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                              columns=feature_names)
    
    # Dự đoán bằng mô hình Random Forest
    prediction = rf_model.predict(input_data)[0]
    probabilities = rf_model.predict_proba(input_data)[0]
    
    st.subheader("📊 Kết quả Chẩn đoán từ Mô hình:")
    
    if prediction == 1:
        risk_prob = probabilities[1] * 100
        st.error(f"⚠️ **CẢNH BÁO: BỆNH NHÂN CÓ NGUY CƠ MẮC Bệnh Tim Mạch!**")
        st.metric(label="Mức độ tin cậy của mô hình (Xác suất mắc bệnh)", value=f"{risk_prob:.2f}%")
        st.write("👉 *Lời khuyên:* Khuyên bệnh nhân nên đến gặp bác sĩ chuyên khoa tim mạch để thực hiện các xét nghiệm chuyên sâu (như chụp mạch vành).")
    else:
        safe_prob = probabilities[0] * 100
        st.success(f"✅ **KẾT QUẢ: AN TOÀN - KHÔNG PHÁT HIỆN DẤU HIỆU BỆNH TIM.**")
        st.metric(label="Mức độ tin cậy của mô hình (Xác suất khỏe mạnh)", value=f"{safe_prob:.2f}%")
        st.write("👉 *Lời khuyên:* Duy trì lối sống lành mạnh, tập thể dục đều đặn và kiểm tra sức khỏe định kỳ.")