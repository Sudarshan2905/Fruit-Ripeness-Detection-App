import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
import tempfile

# ---- Load Model & Encoder ----
rf = joblib.load("rf_ripeness_model.joblib")
le = joblib.load("label_encoder.joblib")

# ---- Feature Extraction Settings ----
LBP_P, LBP_R, LBP_METHOD = 24, 3, "uniform"
HSV_BINS = [8, 8, 8]

def extract_features_from_image(img):
    img = cv2.resize(img, (128, 128))

    # HSV Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # LBP Texture
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method=LBP_METHOD)
    lbp_bins = int(lbp.max() + 1)
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins))
    hist_lbp = hist_lbp.astype("float") / (hist_lbp.sum() + 1e-8)

    return np.hstack([hist, hist_lbp])


# ---- Streamlit UI ----
st.title("Fruit  Ripeness Detection")
st.write("Upload a Fruit image to predict its ripeness stage.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show the image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save image temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

    # Read image with OpenCV
    img = cv2.imread(tmp_path)
    if img is None:
        st.error("Error reading image!")
    else:
        # Extract features
        feats = extract_features_from_image(img).reshape(1, -1)

        # Predict
        probs = rf.predict_proba(feats)[0]
        pred_idx = np.argmax(probs)
        pred_label = le.classes_[pred_idx]
        pred_conf = probs[pred_idx] * 100

        # Verdict Messages
        verdicts = {
            "unripe": "🟢 UNRIPE – Not ready to eat",
            "ripe": "🟡 RIPE – Perfect to eat",
            "overripe": "🟠 OVERRIPE – Use soon",
            "rotten": "🔴 ROTTEN – Do not eat"
        }

        st.subheader(" Prediction Result")
        st.write(f"### {verdicts.get(pred_label, 'Unknown')} ({pred_conf:.2f}% confident)")

        # Probability Table
        st.subheader("📊 Class Probabilities")
        prob_table = {cls: f"{p*100:.2f}%" for cls, p in zip(le.classes_, probs)}
        st.json(prob_table)

        st.success("Prediction completed!")
