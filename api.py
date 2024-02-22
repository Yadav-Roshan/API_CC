from flask import Flask, request, jsonify
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return 'Can you see me?'

def extract_features(image_path):
    model  = pickle.load(open(r"C:\Users\Roshan\OneDrive\Desktop\API\resnet50.pkl", 'rb'))

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array)
    features = features.flatten()

    return features

def pca(features):
    X_pca  = pickle.load(open(r'C:\Users\Roshan\OneDrive\Desktop\API\pca.pkl', 'rb'))
    return X_pca.transform([features])[0]

def normalize(pca_features):
    scaler  = pickle.load(open(r'C:\Users\Roshan\OneDrive\Desktop\API\scaler.pkl', 'rb'))
    return scaler.transform([pca_features])[0]

def predict(normalized_features):
    svm = pickle.load(open(r'C:\Users\Roshan\OneDrive\Desktop\API\svm.pkl', 'rb'))
    prediction = svm.predict(normalized_features.reshape(1, -1))[0]
    if prediction == 0:
        return "Stage 1"
    elif prediction == 1:
        return "Stage 2"
    else:
        return "Stage 3"

@app.route('/predict', methods=['POST'])
def predict_stage():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No image provided'})

    try:
        image_path = r"C:\Users\Roshan\OneDrive\Desktop\temp_image.png"
        image_file.save(image_path)
        features = extract_features(image_path)
        pca_features = pca(features)
        normalized_features = normalize(pca_features)
        prediction = predict(normalized_features)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)