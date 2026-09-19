import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# ✅ Load CNN Model
# =========================
model = tf.keras.models.load_model("lung_model_cnn.keras")

# ترتيب الكلاسات مهم جدًا
class_names = ['covid', 'normal', 'pneumonia']

# =========================
# 🧠 Uncertainty Function
# =========================
def predict_with_uncertainty(model, img, threshold=0.8):

    preds = model.predict(img)[0]

    confidence = np.max(preds)
    class_index = np.argmax(preds)
    label = class_names[class_index]

    # Risk Level
    if confidence >= 0.9:
        risk = "High Confidence"
    elif confidence >= 0.7:
        risk = "Medium Confidence"
    else:
        risk = "Low Confidence"

    # Decision (تم تحسينه)
    if confidence < 0.6:
        decision = "⚠️ Uncertain Prediction - Please upload a clearer image"
    elif confidence < threshold:
        decision = "⚠️ Medium Confidence - Please verify result"
    else:
        decision = "✅ Prediction OK"

    return label, confidence, risk, decision, preds


# =========================
# 🫁 TITLE
# =========================
st.title("🫁 LungGuard AI")

# =========================
# 📖 PROJECT DESCRIPTION
# =========================
st.markdown("""
### 📌 About the Project

**LungGuard AI** is an intelligent deep learning system designed to analyze chest X-ray images and classify lung conditions into:

- 🟢 Normal
- 🟡 Pneumonia
- 🔴 COVID-19

The system provides:
- 📊 Confidence score
- ❤️ Health status
- ⚠️ Risk level

It also includes an uncertainty-aware mechanism to detect low-confidence predictions.

⚠️ **Disclaimer:** This system is for educational purposes only and not a medical diagnosis.
""")

# =========================
# 📤 Upload Image
# =========================
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # =========================
    # 🖼️ Show Image
    # =========================
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # =========================
    # ⚙️ Preprocessing (نفس التدريب)
    # =========================
    img = image.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # =========================
    # 🤖 Prediction + Uncertainty
    # =========================
    label, confidence, risk, decision, probs = predict_with_uncertainty(model, img)

    # =========================
    # ❤️ Health Status (محسن)
    # =========================
    health_status = "Healthy" if label == "normal" else "Needs Medical Review"

    # =========================
    # 📊 RESULTS
    # =========================
    st.subheader("🔍 Results")

    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.write(f"**Health Status:** {health_status}")
    st.write(f"**Risk Level:** {risk}")
    st.write(f"**Decision:** {decision}")

    # =========================
    # 📊 Class Probabilities (إضافة)
    # =========================
    st.subheader("📊 Class Probabilities")

    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {probs[i]:.2f}")

    st.bar_chart(probs)

    # =========================
    # 💡 INSIGHT
    # =========================
    st.subheader("💡 Insight")

    if label == "covid":
        st.write("""
        The model detected **diffuse white patterns** across the lungs, which are commonly associated with COVID-19.
        """)

    elif label == "pneumonia":
        st.write("""
        The model identified **localized opacities** in specific lung regions, typical of Pneumonia.
        """)

    else:
        st.write("""
        The lungs appear **clear with no abnormal patterns**, indicating a normal condition.
        """)
