from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import uvicorn

# =======================================
# 1️⃣ إنشاء التطبيق
# =======================================
app = FastAPI(title="Cancer Severity Prediction API")

# =======================================
# 2️⃣ إعداد CORS
# =======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يسمح بطلبات من أي مصدر
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================
# 3️⃣ تحميل الموديل والـ Scaler والأعمدة
# =======================================
print("⏳ Loading model and scaler ...")
model_columns = joblib.load("model_columns.pkl")     # الأعمدة الأصلية بعد الـ Encoding
scaler = joblib.load("scaler.pkl")                   # StandardScaler
model = joblib.load("regression_cancer.joblib")      # الموديل
print("✅ Model and scaler loaded successfully!")

# =======================================
# 4️⃣ معالجة البيانات قبل التنبؤ
# =======================================
def preprocess_input(data: dict):
    """تحويل الداتا إلى DataFrame ومعالجتها قبل التنبؤ"""
    df = pd.DataFrame([data])

    # تحويل القيم النصية لنوع string لضمان الاتساق
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df)

    # إعادة ترتيب الأعمدة لتطابق التدريب
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    # تطبيق الـ scaler
    df_scaled = scaler.transform(df_encoded)
    return df_scaled

# =======================================
# 5️⃣ Endpoint للتنبؤ
# =======================================
@app.post("/predict_severity")
async def predict_severity(request: Request):
    data = await request.json()

    # معالجة البيانات
    features = preprocess_input(data)

    # التنبؤ
    prediction = model.predict(features)[0]

    # تحديد الفئة
    if prediction < 3:
        category = "Low"
    elif prediction < 6:
        category = "Medium"
    else:
        category = "High"

    return {
        "predicted_score": float(prediction),
        "severity_category": category
    }

# =======================================
# 6️⃣ الصفحة الرئيسية
# =======================================
@app.get("/")
def root():
    return {"message": "Welcome to the Cancer Severity Prediction API 🚀"}

# =======================================
# 7️⃣ تشغيل السيرفر
# =======================================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
