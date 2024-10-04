from flask import Flask, render_template, request, redirect, url_for
# import cv2 as cv
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.preprocessing import image
from keras.models import load_model
# from PIL import Image

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

# URLs for the models stored in your GitHub repo
resnet50_url = 'https://raw.githubusercontent.com/bizzle22/Alopecia-Detection/main/models/resnet50_model.h5'
svm_cnn_url = 'https://raw.githubusercontent.com/bizzle22/Alopecia-Detection/main/models/svm_using_cnn_smote.h5'
vgg19_url = 'https://raw.githubusercontent.com/bizzle22/Alopecia-Detection/main/models/vgg19_SMOTE_ksh.h5'

# Define local paths where models will be saved
models_dir = './models'
os.makedirs(models_dir, exist_ok=True)

resnet50_path = os.path.join(models_dir, 'resnet50_model.h5')
svm_cnn_path = os.path.join(models_dir, 'svm_using_cnn_smote.h5')
vgg19_path = os.path.join(models_dir, 'vgg19_SMOTE_ksh.h5')

# Function to download and save the models
def download_model(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {local_path}...")
        with open(local_path, 'wb') as f:
            response = requests.get(url)
            f.write(response.content)
        print(f"Downloaded and saved to {local_path}")
    else:
        print(f"{local_path} already exists.")

# Download the models if not already downloaded
download_model(resnet50_url, resnet50_path)
download_model(svm_cnn_url, svm_cnn_path)
download_model(vgg19_url, vgg19_path)

# Function to perform prediction
# Function to perform prediction
def predict(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array /= 255.
    img_array = np.expand_dims(img_array, axis=0)

    # Load models
    resnet50_model = load_model(resnet50_path)
    svm_cnn_model = load_model(svm_cnn_path)
    vgg19_model = load_model(vgg19_path)

    # Make predictions
    resnet50_prediction = resnet50_model.predict(img_array)
    svm_cnn_prediction = svm_cnn_model.predict(img_array)
    vgg19_prediction = vgg19_model.predict(img_array)

    # Interpret predictions (assuming binary classification with threshold 0.5)
    resnet50_result = 'Alopecia' if resnet50_prediction < 0.5 else 'Healthy Hair'
    svm_cnn_result = 'Alopecia' if svm_cnn_prediction < 0.5 else 'Healthy Hair'
    vgg19_result = 'Alopecia' if vgg19_prediction < 0.5 else 'Healthy Hair'

    # Ensembling (majority vote)
    ensemble_prediction = [resnet50_result, svm_cnn_result, vgg19_result]
    final_prediction = 'Alopecia Detected' if ensemble_prediction.count('Alopecia') >= 2 else 'Healthy Hair'

    return final_prediction

@app.route('/Imageupload.html', methods=['GET', 'POST'])
def image_upload():
    if request.method == 'POST':
        img = request.files['Scalp']
        img.save('uploaded_image.jpg')
        return redirect(url_for('result'))
    return render_template('Imageupload.html')

@app.route('/result')
def result():
    prediction = predict('uploaded_image.jpg')
    return render_template('result.html', prediction=prediction)

