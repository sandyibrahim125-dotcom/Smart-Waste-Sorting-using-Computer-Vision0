"""
Smart Waste Sorting — Streamlit App
تصنيف المخلفات الذكي — تطبيق Streamlit

Loads the model exported from the training notebook and classifies an uploaded
or captured photo into one of: cardboard, glass, metal, paper, plastic, trash —
then shows the recommended recycling/treatment method for that material.
"""

import streamlit as st
from PIL import Image

from model.predict import inference

st.set_page_config(
    page_title="Smart Waste Sorting",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ Smart Waste Sorting using Computer Vision")
st.caption("تصنيف المخلفات الذكي باستخدام الرؤية الحاسوبية")

st.write(
    "Upload a photo, or take one with your camera, and the model will classify "
    "the item and suggest how it should be recycled or processed."
)

input_method = st.radio(
    "Choose input method / اختر طريقة الإدخال",
    ["Upload image", "Use camera"],
    horizontal=True,
)

image = None
if input_method == "Upload image":
    uploaded_file = st.file_uploader(
        "Upload an image of the waste item", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Take a photo of the waste item")
    if camera_file is not None:
        image = Image.open(camera_file)

if image is not None:
    st.image(image, caption="Input image", use_container_width=True)

    with st.spinner("Classifying..."):
        try:
            result = inference(image)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    st.subheader(f"Prediction: **{result['label'].capitalize()}**")
    st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

    st.write("#### All class probabilities")
    st.bar_chart(result["probabilities"])

    guidance = result.get("guidance", {})
    if guidance:
        st.write("#### ♻️ Recommended treatment / كيفية المعالجة")
        st.info(f"**EN:** {guidance.get('en', '')}")
        st.info(f"**AR:** {guidance.get('ar', '')}")
else:
    st.info("Upload or capture an image to get a prediction.")

st.divider()
st.caption(
    "Model: MobileNetV2 transfer learning, trained on TrashNet-style data "
    "(cardboard, glass, metal, paper, plastic, trash). "
    "See the accompanying notebook for the full training pipeline."
)
