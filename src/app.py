# src/app.py
import streamlit as st
from pathlib import Path
from PIL import Image
import torch, sys, io, os
import numpy as np
from typing import Tuple, List

# ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from model import get_model
from utils import preprocess_image, load_checkpoint

# ---------- CONFIG ----------
MODEL_PATH = ROOT / "saved_models" / "best_model.pth"
BEST_THRESH = 0.0914    # optional: keep if you have a threshold; else ignore
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Tuberculosis Detection — Chest X-rays",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Simple CSS ----------
st.markdown(
    """
    <style>
    .big-title {font-size:30px; font-weight:700; color:#0b5460; margin-bottom:6px;}
    .subtitle {font-size:13px; color:#4b5b5f; margin-top:0px; margin-bottom:10px;}
    .small-muted {color: #666; font-size:12px;}
    .result-box {padding:14px; border-radius:8px;}
    .card {background:#fff; padding:12px; border-radius:8px; box-shadow:0 4px 12px rgba(0,0,0,0.06);}
    .probbar {height:10px; background:#eef6f8; border-radius:6px; overflow:hidden;}
    .probfill {height:10px; background:linear-gradient(90deg,#1e90ff,#0066cc); border-radius:6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
@st.cache_resource
def load_model_cached(path: str = str(MODEL_PATH)):
    """Load PyTorch model checkpoint. Returns dict with model, classes, device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(num_classes=2, pretrained=False)
    if Path(path).exists():
        try:
            model, classes = load_checkpoint(path, model, device=device)
            model.to(device).eval()
            return {"ok": True, "model": model, "classes": classes, "device": device}
        except Exception as e:
            return {"ok": False, "error": str(e), "device": device}
    else:
        return {"ok": False, "error": "checkpoint not found", "device": device}

def predict_from_pil(img_pil: Image.Image, model_meta) -> Tuple[int, float, List[float]]:
    """Return (pred_idx, pred_confidence, probs_list)."""
    device = model_meta["device"]
    model = model_meta["model"]
    # reuse utils.preprocess_image to ensure same transforms
    t = preprocess_image(img_pil).to(device)
    with torch.no_grad():
        out = model(t)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0].tolist()
        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])
    return pred_idx, pred_conf, probs

def pil_from_uploaded(uploaded) -> Image.Image:
    if isinstance(uploaded, (str, Path)):
        return Image.open(uploaded).convert("RGB")
    if hasattr(uploaded, "read"):
        return Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    return Image.open(uploaded).convert("RGB")

# ---------- UI ----------
st.markdown('<div class="big-title">🫁 Tuberculosis Detection — Chest X-rays</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a frontal chest X-ray (JPG/PNG). Model predicts <b>Normal</b> or <b>Tuberculosis</b> with confidence and guidance.</div>', unsafe_allow_html=True)
st.markdown("---")

# Load model
model_meta = load_model_cached()

if not model_meta["ok"]:
    if "error" in model_meta:
        st.warning(f"Model not ready: {model_meta.get('error')}")
    else:
        st.warning("Model not ready. Place checkpoint at: saved_models/best_model.pth")
    # but continue to allow user to upload image and see message

# Layout
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Upload X-ray")
    st.markdown("Drag & drop or click to browse. Supported: JPG, JPEG, PNG.")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

    # optional sample preview area
    sample_folder = ROOT / "data" / "sample"
    if sample_folder.exists():
        st.markdown("**Try sample images:**")
        sample_files = sorted([p for p in sample_folder.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])[:3]
        scols = st.columns(max(1, len(sample_files)))
        for p, c in zip(sample_files, scols):
            with c:
                try:
                    im = Image.open(p).convert("RGB")
                    st.image(im.resize((120,120)), use_column_width=False)
                    if st.button(f"Use {p.name}", key=str(p)):
                        uploaded = p
                except Exception:
                    pass
    else:
        st.info("Put example images in data/sample/ to show sample thumbnails here.")

    if uploaded:
        try:
            image = pil_from_uploaded(uploaded)
            st.markdown("**Preview:**")
            st.image(image, use_column_width=True)
            # keep bytes for download if needed
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_bytes = buf.getvalue()
        except Exception as e:
            st.error("Could not read uploaded image. Try another file.")
            image = None
    else:
        st.info("Upload a chest X-ray to run inference.")
        image = None

with right:
    st.subheader("Prediction & Guidance")
    if image is None:
        st.info("Prediction will appear here after you upload an image.")
    else:
        if not model_meta["ok"]:
            st.warning("Model not loaded. Cannot run inference.")
        else:
            if st.button("Predict"):
                with st.spinner("Running model inference..."):
                    try:
                        pred_idx, pred_conf, probs = predict_from_pil(image, model_meta)
                        classes = model_meta.get("classes") or ["Normal", "TB"]
                        tb_index = 1 if len(classes) > 1 else 0  # assuming index 1 corresponds to TB class saved earlier
                        tb_prob = probs[tb_index]
                        normal_prob = probs[1 - tb_index] if len(probs) > 1 else 1 - tb_prob
                        # decision using threshold (optional)
                        result_flag = 1 if tb_prob > BEST_THRESH else 0

                        # show result box
                        st.markdown("<div class='result-box card'>", unsafe_allow_html=True)
                        if result_flag == 1:
                            # TB predicted
                            if tb_prob >= 0.90:
                                st.error(f"⚠️ **Prediction:** Tuberculosis detected  \n**Confidence (TB):** {tb_prob:.2%}  \n**Interpretation:** High confidence")
                            elif tb_prob >= 0.70:
                                st.warning(f"🟡 **Prediction:** Tuberculosis suspected  \n**Confidence (TB):** {tb_prob:.2%}  \n**Interpretation:** Medium confidence — consider further tests")
                            else:
                                st.info(f"ℹ️ **Prediction:** Tuberculosis (Uncertain)  \n**Confidence (TB):** {tb_prob:.2%}  \n**Interpretation:** Low confidence — retest/refer to clinician")
                        else:
                            # Normal predicted
                            if normal_prob >= 0.90:
                                st.success(f"✅ **Prediction:** Normal (No signs of TB)  \n**Confidence (Normal):** {normal_prob:.2%}  \n**Interpretation:** High confidence")
                            elif normal_prob >= 0.70:
                                st.warning(f"🟡 **Prediction:** Likely Normal  \n**Confidence (Normal):** {normal_prob:.2%}  \n**Interpretation:** Medium confidence")
                            else:
                                st.error(f"⚠️ **Prediction:** Uncertain / Inconclusive  \n**Confidence (Normal):** {normal_prob:.2%}  \n**Interpretation:** Low confidence — retest or seek expert opinion")
                        st.markdown("</div>", unsafe_allow_html=True)

                        # details & metrics
                        st.markdown("---")
                        st.markdown(f"**Model threshold used:** {BEST_THRESH:.4f}")
                        st.markdown(f"**Raw TB probability:** {tb_prob:.4f}")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("TB Probability", f"{tb_prob:.2%}")
                        c2.metric("Normal Probability", f"{normal_prob:.2%}")
                        c3.metric("Threshold", f"{BEST_THRESH:.3f}")

                        with st.expander("How to interpret this result"):
                            st.markdown("""
                            - **High (≥90%)**: Model confident. Confirm clinically.  
                            - **Medium (70–90%)**: Consider further tests or expert review.  
                            - **Low (<70%)**: Uncertain — retest or consult clinician.
                            - **Note:** This is a prototype, not a clinical diagnostic tool.
                            """)
                        # Download simple txt report
                        report_txt = (
                            f"Prediction: {'Tuberculosis' if result_flag==1 else 'Normal'}\n"
                            f"TB probability (model): {tb_prob:.6f}\n"
                            f"Normal probability: {normal_prob:.6f}\n"
                            f"Threshold used: {BEST_THRESH:.6f}\n"
                        )
                        st.download_button("⬇️ Download simple report (txt)", report_txt, file_name="tb_report.txt", mime="text/plain")
                    except Exception as e:
                        st.error("Inference failed: " + str(e))

st.markdown("---")
colf1, colf2 = st.columns([3,1])
with colf1:
    st.markdown("**Author:** Youraj Kumar  •  **Project:** Tuberculosis Detection using Deep Learning")
    st.markdown("GitHub: add-your-repo-link  •  Dataset: Kaggle `Tuberculosis Chest X-rays Images`")
with colf2:
    st.markdown("v1.0")
