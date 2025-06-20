import streamlit as st
import os
import pickle
import numpy as np
import skimage.io
import skimage.transform
import skimage.color
import skimage.feature
import scipy.stats
import scipy.special

st.set_page_config(page_title="Image Classifier", layout="centered")

# Paths for model files (assume static/models location in deploy)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "static", "models")
SGD_MODEL_FILE = os.path.join(
    MODEL_PATH, "dsa_image_classification_sgd.pickle")
SCALER_FILE = os.path.join(MODEL_PATH, "dsa_scaler.pickle")

# Load model & scaler


@st.cache_resource
def load_model():
    model_sgd = pickle.load(open(SGD_MODEL_FILE, "rb"))
    scaler = pickle.load(open(SCALER_FILE, "rb"))
    return model_sgd, scaler


model_sgd, scaler = load_model()

st.title("Image Classifier App")
st.write("Upload an image (.jpg, .jpeg, .png) to get the top-5 predicted labels.")

with st.sidebar:
    st.info("About: This app classifies uploaded images using a pre-trained SGD classifier. Built with Streamlit.")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png"])

# Helper for resizing to display (mimics getheight)


def getheight(image):
    h, w, _ = image.shape
    aspect = h / w
    given_width = 300
    height = given_width * aspect
    return int(height)

# Pipeline logic (adapted from Flask)


def pipeline_model(image_data, scaler_transform, model_sgd):
    image = skimage.io.imread(image_data)
    # Resize to 80x80
    image_resize = skimage.transform.resize(image, (80, 80))
    image_scale = 255 * image_resize
    image_transform = image_scale.astype(np.uint8)
    gray = skimage.color.rgb2gray(image_transform)
    feature_vector = skimage.feature.hog(
        gray, orientations=10, pixels_per_cell=(8, 8), cells_per_block=(2, 2)
    )
    scalex = scaler_transform.transform(feature_vector.reshape(1, -1))
    result = model_sgd.predict(scalex)
    # Confidence
    decision_value = model_sgd.decision_function(scalex).flatten()
    labels = model_sgd.classes_
    z = scipy.stats.zscore(decision_value)
    prob_value = scipy.special.softmax(z)
    # Top 5
    top_5_prob_ind = prob_value.argsort()[::-1][:5]
    top_labels = labels[top_5_prob_ind]
    top_prob = prob_value[top_5_prob_ind]
    top_dict = {k: float(np.round(v, 3)) for k, v in zip(top_labels, top_prob)}
    return top_dict


if uploaded_file is not None:
    st.image(uploaded_file, width=300, caption="Uploaded Image")
    # Save/uploaded file to temp for skimage
    img_np = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file.write(img_np)
        tmp_path = tmp_file.name
    try:
        results = pipeline_model(tmp_path, scaler, model_sgd)
        st.success("Top-5 Predictions:")
        st.table([(k, v) for k, v in results.items()])
    except Exception as e:
        st.error(f"Error: {e}")
    os.remove(tmp_path)  # Clean up tmpfile
else:
    st.info("Please upload an image file.")
