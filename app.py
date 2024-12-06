import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from train_model import DṛṣṭiGyanCNN  # Import the custom model

# Load the trained model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the custom CNN model
    model = DṛṣṭiGyanCNN(num_classes=100)

    # Load the state dictionary
    state_dict = torch.load('./DṛṣṭiGyan v1.pth', map_location=device)  # Replace with your model path if necessary

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Move the model to the appropriate device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    return model, device

# Preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Predict function
def predict_image(model, device, image):
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Class names for CIFAR-100
def get_class_names():
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
        'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
        'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
        'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
        'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
        'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
        'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
        'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

# Load model and device
model, device = load_model()

# Streamlit App UI
st.set_page_config(page_title="DṛṣṭiGyan - AI-Powered Vision Knowledge", page_icon="🎨", layout="centered")
st.markdown(
    """
    <style>
    .title {
        font-size: 2rem;
        color: #FF5733;
        font-weight: bold;
        text-align: center;
    }
    .footer {
        font-size: 1rem;
        color: #6C3483;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>Welcome to <span style='color:#2E86C1;'>DṛṣṭiGyan</span> 🎨</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Developed by Shivam AKA <strong>💥💪Bajrang Bhakt Shivam💪💥</strong></p>", unsafe_allow_html=True)

# File uploader or camera input
uploaded_file = st.file_uploader("Upload an Image or Use Camera", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Or click an image directly")

if uploaded_file or camera_image:
    st.write("### Uploaded Image")
    image = Image.open(uploaded_file if uploaded_file else camera_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the class
    prediction = predict_image(model, device, processed_image)

    # Get class names
    class_names = get_class_names()

    # Display the prediction
    st.success(f"Predicted Class: **{class_names[prediction]}**")

st.markdown(
    """
    <div class='footer'>
        <p>Powered by <strong>DṛṣṭiGyan</strong>. Built with ❤️ by Shivam AKA 👊💪Bajrang Bhakt Shivam💪👊.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
