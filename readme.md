Streamlit Image Classifier App
A user-friendly web app for classifying images into categories using a pre-trained SGD Classifier. Built with Python and Streamlit.
Easily upload images and instantly see the top-5 predicted labels with confidence scores.
Features
Upload .jpg, .jpeg, or .png images
Preprocessing and feature extraction (HOG) built-in
Displays top-5 prediction results with probabilities
Fast and intuitive interface
Demo
Demo Screenshot
Update with your own image if desired
Getting Started
Prerequisites
Python 3.9 or newer recommended
Model files: dsa_image_classification_sgd.pickle and dsa_scaler.pickle inside static/models/
Installation
Clone this repository:

Install requirements:

Add your model files:
Place dsa_image_classification_sgd.pickle and dsa_scaler.pickle in static/models/.
Run the App

Visit http://localhost:8501 in your browser.
Deploying to Streamlit Cloud
Push your project to GitHub.
Go to Streamlit Cloud.
Click “New app” and select your repository.
Set the main file as streamlit_app.py and click ‘Deploy’.
Enjoy your app online!
File Structure

Acknowledgements
Based on original Flask project
Powered by scikit-learn, scikit-image, and Streamlit