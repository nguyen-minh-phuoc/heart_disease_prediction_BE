"""
Cardiovascular Disease Prediction API
Backend API sử dụng FastAPI và CatBoost model
Dataset: cleaned_cardio_train.csv với feature engineering
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
from typing import Dict
import uvicorn

# Khởi tạo FastAPI app
app = FastAPI(
    title="Cardiovascular Disease Prediction API",
    description="API dự đoán bệnh tim mạch dựa trên thông tin sức khỏe bệnh nhân",
    version="2.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model CatBoost
try:
    model = CatBoostClassifier()
    model.load_model("model_catboost_production.cbm")
    print("✅ Đã load model CatBoost thành công!")
except Exception as e:
    print(f"❌ Lỗi khi load model: {e}")
    model = None


def feature_engineering(data: dict) -> pd.DataFrame:
    """
    Feature Engineering giống như trong training code
    Tạo các features mới từ dữ liệu đầu vào
    """
    # Tạo DataFrame từ input
    df = pd.DataFrame([data])
    
    # Tính tuổi theo ngày (nếu chưa có)
    df['age'] = df['age_years'] * 365
    
    # BMI
    df['bmi'] = df['weight'] / (df['height'] / 100)**2
    
    # Pulse Pressure
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # Mean Arterial Pressure
    df['map'] = (2 * df['ap_lo'] + df['ap_hi']) / 3
    
    # Age decade
    df['age_decade'] = (df['age_years'] // 10).astype(int)
    
    # BMI groups
    df['bmi_group'] = pd.cut(df['bmi'], 
                             bins=[0, 18.5, 25, 30, 35, 100],
                             labels=['underweight', 'normal', 'overweight', 'obese', 'extreme_obese'])
    
    # Blood Pressure stages
    df['bp_stage'] = pd.cut(df['ap_hi'], 
                            bins=[0, 119, 129, 139, 159, 179, 300],
                            labels=['normal', 'elevated', 'stage1', 'stage2', 'stage3', 'crisis'])
    
    # Pulse Pressure category
    df['pp_category'] = pd.cut(df['pulse_pressure'], 
                               bins=[0, 40, 60, 100],
                               labels=['normal', 'high', 'very_high'])
    
    # MAP category
    df['map_category'] = pd.cut(df['map'], 
                                bins=[0, 93, 107, 200],
                                labels=['low', 'normal', 'high'])
    
    # Health score (tổng điểm sức khỏe)
    df['health_score'] = (
        (df['cholesterol'] == 1).astype(int) +
        (df['gluc'] == 1).astype(int) +
        (df['smoke'] == 0).astype(int) +
        (df['alco'] == 0).astype(int) +
        (df['active'] == 1).astype(int)
    )
    
    # Risk factors (số lượng yếu tố nguy cơ)
    df['risk_factors'] = (
        (df['smoke'] == 1).astype(int) +
        (df['alco'] == 1).astype(int) +
        (df['active'] == 0).astype(int) +
        (df['cholesterol'] >= 2).astype(int) +
        (df['gluc'] >= 2).astype(int)
    )
    
    # Interaction features
    df['age_bmi'] = df['age_years'] * df['bmi']
    df['age_ap_hi'] = df['age_years'] * df['ap_hi']
    df['bmi_ap_hi'] = df['bmi'] * df['ap_hi']
    
    # Polynomial features
    for var in ['age_years', 'bmi', 'ap_hi', 'ap_lo']:
        df[f'{var}_squared'] = df[var] ** 2
    
    # Ratio features
    df['weight_height_ratio'] = df['weight'] / df['height']
    df['systolic_diastolic_ratio'] = df['ap_hi'] / df['ap_lo']
    
    # Convert categorical columns to string for CatBoost
    cat_cols = ['gender', 'bmi_group', 'bp_stage', 'pp_category', 'map_category']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df


# Schema cho dữ liệu đầu vào
class PatientData(BaseModel):
    """Thông tin bệnh nhân đầu vào (11 features cơ bản)"""
    age_years: float = Field(..., ge=1, le=120, description="Tuổi (năm)")
    gender: int = Field(..., ge=1, le=2, description="Giới tính (1: Nữ, 2: Nam)")
    height: float = Field(..., ge=130, le=210, description="Chiều cao (cm)")
    weight: float = Field(..., ge=30, le=200, description="Cân nặng (kg)")
    ap_hi: int = Field(..., ge=80, le=200, description="Huyết áp tâm thu (mmHg)")
    ap_lo: int = Field(..., ge=40, le=130, description="Huyết áp tâm trương (mmHg)")
    cholesterol: int = Field(..., ge=1, le=3, description="Cholesterol (1: Bình thường, 2: Cao, 3: Rất cao)")
    gluc: int = Field(..., ge=1, le=3, description="Glucose (1: Bình thường, 2: Cao, 3: Rất cao)")
    smoke: int = Field(..., ge=0, le=1, description="Hút thuốc (0: Không, 1: Có)")
    alco: int = Field(..., ge=0, le=1, description="Uống rượu (0: Không, 1: Có)")
    active: int = Field(..., ge=0, le=1, description="Hoạt động thể chất (0: Không, 1: Có)")

    class Config:
        json_schema_extra = {
            "example": {
                "age_years": 55,
                "gender": 2,
                "height": 168,
                "weight": 75,
                "ap_hi": 140,
                "ap_lo": 90,
                "cholesterol": 2,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1
            }
        }


# Schema cho kết quả dự đoán
class PredictionResult(BaseModel):
    """Kết quả dự đoán"""
    prediction: int  # 0: Không bệnh, 1: Có bệnh
    probability: float  # Xác suất mắc bệnh (0-1)
    risk_level: str  # Mức độ nguy cơ
    message: str  # Thông báo chi tiết
    bmi: float  # BMI để thông tin thêm


def get_risk_level(probability: float) -> str:
    """Xác định mức độ nguy cơ dựa trên xác suất"""
    if probability < 0.3:
        return "Thấp"
    elif probability < 0.5:
        return "Trung bình thấp"
    elif probability < 0.7:
        return "Trung bình"
    elif probability < 0.85:
        return "Cao"
    else:
        return "Rất cao"


def get_message(prediction: int, probability: float, risk_level: str, bmi: float, patient_data: dict) -> str:
    """Tạo thông báo chi tiết cho bệnh nhân"""
    bmi_status = "bình thường"
    if bmi < 18.5:
        bmi_status = "thiếu cân"
    elif bmi >= 30:
        bmi_status = "béo phì"
    elif bmi >= 25:
        bmi_status = "thừa cân"
    
    # ============================================================
    # YẾU TỐ NGUY CƠ CHÍNH (theo Feature Importance của model)
    # ============================================================
    risk_factors_list = []
    
    # 1. Huyết áp cao - Feature quan trọng nhất
    ap_hi = patient_data.get('ap_hi', 0)
    ap_lo = patient_data.get('ap_lo', 0)
    if ap_hi > 140 or ap_lo > 90:
        risk_factors_list.append(f"Huyết áp cao ({ap_hi}/{ap_lo} mmHg)")
    
    # 2. Tuổi cao - Feature quan trọng thứ 2 (format integer, không thập phân)
    age_years = int(patient_data.get('age_years', 0))
    if age_years > 55:
        risk_factors_list.append(f"Tuổi cao ({age_years} tuổi)")
    
    # 3. BMI cao - Feature quan trọng
    if bmi >= 30:
        risk_factors_list.append(f"Béo phì (BMI {bmi:.1f})")
    elif bmi >= 27:
        risk_factors_list.append(f"Thừa cân nhiều (BMI {bmi:.1f})")
    
    # 4. Cholesterol cao - Feature quan trọng
    cholesterol = patient_data.get('cholesterol', 1)
    if cholesterol == 3:
        risk_factors_list.append("Cholesterol rất cao")
    elif cholesterol == 2:
        risk_factors_list.append("Cholesterol cao")
    
    # 5. Glucose cao - Feature quan trọng
    gluc = patient_data.get('gluc', 1)
    if gluc == 3:
        risk_factors_list.append("Glucose rất cao (nguy cơ tiểu đường)")
    elif gluc == 2:
        risk_factors_list.append("Glucose cao")
    
    # ============================================================
    # THÓI QUEN SINH HOẠT (để đưa ra khuyến nghị)
    # ============================================================
    lifestyle_issues = []
    if patient_data.get('smoke') == 1:
        lifestyle_issues.append("hút thuốc lá")
    if patient_data.get('alco') == 1:
        lifestyle_issues.append("uống rượu/bia")
    if patient_data.get('active') == 0:
        lifestyle_issues.append("ít hoạt động thể chất")
    
    # ============================================================
    # TẠO MESSAGE
    # ============================================================
    base_message = f"Mức độ nguy cơ: {risk_level} ({probability*100:.1f}%)\nBMI: {bmi:.1f} ({bmi_status})"
    
    # Thêm yếu tố nguy cơ chính (nếu có)
    if len(risk_factors_list) > 0:
        risk_text = f"\n\nYếu tố nguy cơ:\n• " + "\n• ".join(risk_factors_list)
        base_message += risk_text
    
    # ============================================================
    # KHUYẾN NGHỊ
    # ============================================================
    if prediction == 0:
        # KHÔNG có dấu hiệu bệnh
        if len(risk_factors_list) == 0 and len(lifestyle_issues) == 0:
            # Hoàn toàn khỏe mạnh
            recommendation = "\n\nKhuyến nghị:\nTình trạng sức khỏe tốt. Tiếp tục duy trì lối sống lành mạnh và khám sức khỏe định kỳ 6-12 tháng/lần."
        
        elif len(risk_factors_list) > 0 and len(lifestyle_issues) == 0:
            # Có yếu tố nguy cơ nhưng lối sống tốt
            recommendation = "\n\nKhuyến nghị:\nMặc dù chưa có dấu hiệu bệnh, cần theo dõi và kiểm soát các yếu tố nguy cơ trên. Khám sức khỏe định kỳ 3-6 tháng/lần."
        
        elif len(risk_factors_list) == 0 and len(lifestyle_issues) > 0:
            # Lối sống không tốt nhưng chưa có yếu tố nguy cơ
            recommendation = f"\n\nKhuyến nghị:\nHiện tại chưa có dấu hiệu bệnh, nhưng nên cải thiện thói quen: {', '.join(lifestyle_issues)}. Những thói quen này có thể gia tăng nguy cơ bệnh tim mạch trong tương lai."
        
        else:
            # Cả yếu tố nguy cơ VÀ lối sống không tốt
            recommendation = f"\n\nKhuyến nghị:\nNên cải thiện thói quen: {', '.join(lifestyle_issues)}. Kết hợp với kiểm soát các yếu tố nguy cơ để giảm thiểu nguy cơ phát triển bệnh tim mạch. Khám sức khỏe định kỳ 3-6 tháng/lần."
    
    else:
        # CÓ dấu hiệu bệnh
        if len(lifestyle_issues) > 0:
            recommendation = f"\n\nKhuyến nghị:\nNên gặp bác sĩ chuyên khoa tim mạch để được thăm khám và điều trị. Ngoài ra cần cải thiện thói quen: {', '.join(lifestyle_issues)}."
        else:
            recommendation = "\n\nKhuyến nghị:\nNên gặp bác sĩ chuyên khoa tim mạch để được thăm khám, tư vấn và điều trị kịp thời."
    
    return base_message + recommendation


@app.get("/")
def root():
    """Endpoint gốc - thông tin API"""
    return {
        "message": "Cardiovascular Disease Prediction API",
        "version": "2.0.0",
        "model": "CatBoost",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Kiểm tra tình trạng API và model"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "CatBoost Classifier"
    }


@app.post("/predict", response_model=PredictionResult)
def predict_cardiovascular_disease(patient: PatientData):
    """
    Dự đoán khả năng mắc bệnh tim mạch cho bệnh nhân
    
    Input features (11 features cơ bản):
    - age_years: Tuổi (năm)
    - gender: Giới tính (1: Nữ, 2: Nam)
    - height: Chiều cao (cm)
    - weight: Cân nặng (kg)
    - ap_hi: Huyết áp tâm thu (mmHg)
    - ap_lo: Huyết áp tâm trương (mmHg)
    - cholesterol: Mức cholesterol (1-3)
    - gluc: Mức glucose (1-3)
    - smoke: Hút thuốc (0/1)
    - alco: Uống rượu (0/1)
    - active: Hoạt động thể chất (0/1)
    
    Returns:
    - prediction: 0 (không bệnh) hoặc 1 (có bệnh)
    - probability: Xác suất mắc bệnh (0-1)
    - risk_level: Mức độ nguy cơ
    - message: Thông báo chi tiết
    - bmi: Chỉ số BMI
    - risk_factors_count: Số yếu tố nguy cơ
    """
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load. Vui lòng kiểm tra file model.")
    
    try:
        # Convert input to dict
        patient_dict = patient.dict()
        
        # Feature engineering
        df_features = feature_engineering(patient_dict)
        
        # Get categorical feature indices
        cat_cols = ['gender', 'bmi_group', 'bp_stage', 'pp_category', 'map_category']
        cat_idx = [df_features.columns.get_loc(c) for c in cat_cols if c in df_features.columns]
        
        # Create Pool for CatBoost
        pool = Pool(df_features, cat_features=cat_idx)
        
        # Dự đoán
        prediction = int(model.predict(pool)[0])
        probability = float(model.predict_proba(pool)[0][1])
        
        # Lấy thông tin bổ sung
        bmi = float(df_features['bmi'].values[0])
        
        # Xác định mức độ nguy cơ
        risk_level = get_risk_level(probability)
        
        # Tạo thông báo (truyền patient_dict để phân tích yếu tố nguy cơ)
        message = get_message(prediction, probability, risk_level, bmi, patient_dict)
        
        return PredictionResult(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            message=message,
            bmi=round(bmi, 1)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán: {str(e)}")


@app.get("/model-info")
def get_model_info():
    """Lấy thông tin về model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model chưa được load")
    
    return {
        "model_type": "CatBoost Classifier",
        "base_features": [
            "age_years", "gender", "height", "weight", 
            "ap_hi", "ap_lo", "cholesterol", "gluc", 
            "smoke", "alco", "active"
        ],
        "num_base_features": 11,
        "engineered_features": [
            "bmi", "pulse_pressure", "map", "age_decade",
            "bmi_group", "bp_stage", "pp_category", "map_category",
            "health_score", "risk_factors", "age_bmi", "age_ap_hi",
            "bmi_ap_hi", "squared features", "ratio features"
        ],
        "classes": [0, 1],
        "class_names": ["Không bệnh", "Có bệnh"],
        "roc_auc_score": "~0.80"
    }


if __name__ == "__main__":
    # Chạy server
    print("\n" + "="*70)
    print("  CARDIOVASCULAR DISEASE PREDICTION API")
    print("="*70)
    print("\nStarting server...")
    print("API URL: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )