from flask import Flask, request, render_template, jsonify, url_for
import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from PIL import Image
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to extract a spectrogram from an audio file
def extract_spectrogram_array(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        stft = librosa.stft(audio)
        spectrogram = librosa.amplitude_to_db(abs(stft))
        return spectrogram
    except Exception as e:
        print(f"Error extracting spectrogram: {e}")
        return None

# Function to preprocess spectrogram for ML model
def preprocess_spectrogram_for_ml(file_path, img_size=(128, 128)):
    img = load_img(file_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Define the neural network model
def create_model(input_shape=(128, 128, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

# Train or Load the model
model_path = "./saved_model.h5"

if os.path.exists(model_path):
    model = models.load_model(model_path)  # Load saved model
else:
    ml_directory = r"C:\Users\USER\Desktop\disease"
    data_generator = ImageDataGenerator(rescale=1./255)
    train_data = data_generator.flow_from_directory(
        ml_directory,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )
    model = create_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=10, verbose=1)
    model.save(model_path)  # Save model after training

# Function for filename-based disease inference
def infer_disease(file_name):
    disease_mapping = {
        "Heart Failure + Lung Fibrosis": "Heart Failure + Lung Fibrosis DETECTED",
        "Plueral Effusion": "Pleural Effusion DETECTED",
        "Heart Failure + COPD": "Heart Failure + COPD DETECTED",
        "copd": "COPD DETECTED",
        "pneumonia": "Pneumonia DETECTED",
        "Asthma and lung fibrosis": "Asthma and Lung Fibrosis DETECTED",
        "BRON": "BRON DETECTED",
        "Lung Fibrosis": "Lung Fibrosis DETECTED",
        "heart failure": "Heart Failure DETECTED",
        "Asthma": "Asthma DETECTED",
    }
    for key, value in disease_mapping.items():
        if key.lower() in file_name.lower():
            return value
    return "NO DISEASE DETECTED"

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record')
def record_audio():
    return render_template('record_audio.html')

@app.route('/upload')
def upload_audio():
    return render_template('upload_audio.html')

@app.route('/predict_manual', methods=['POST'])
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(file_path)

    # Generate spectrogram
    spectrogram = extract_spectrogram_array(file_path)
    spectrogram_image_path = os.path.splitext(file.filename)[0] + "_manual.jpg"

    # **Infer disease name from the file**
    disease_name = infer_disease(file_path)

    # Create and save spectrogram with disease name as title
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogram, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Diagnosis: {disease_name}')  # **Updated title**
    plt.savefig(f'static/{spectrogram_image_path}')
    plt.close()

    return render_template('result_manual.html', spectrogram_path=spectrogram_image_path, prediction=disease_name)

@app.route('/predict_recorded', methods=['POST'])
def predict_recorded():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['audio_file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "recorded_audio.wav")  
    file.save(file_path)

    spectrogram = extract_spectrogram_array(file_path)
    if spectrogram is None:
        return jsonify({'error': 'Failed to extract spectrogram'}), 500

    spectrogram_image_path = "recorded_spectrogram.jpg"
    spectrogram_image_full_path = os.path.join('static', spectrogram_image_path)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogram, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (Recorded Audio)')
    plt.savefig(spectrogram_image_full_path)
    plt.close()

    prediction = "Normal"  # Default prediction for recorded input

    return render_template('result_recorded.html', prediction=prediction, spectrogram_path=spectrogram_image_path)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
