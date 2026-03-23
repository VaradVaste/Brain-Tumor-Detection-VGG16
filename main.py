from flask import Flask, render_template, request, send_from_directory, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
import os

# ========================
# App Setup
# ========================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========================
# Load Model
# ========================
model = load_model(os.path.join("models", "Brain_Tumor.h5"))

class_labels = ['pituitary', 'notumor', 'meningioma', 'glioma']


# ========================
# Utility Functions
# ========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    IMAGE_SIZE = 128

    img = tf.io.read_file(file_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = img / 255.0

    return tf.expand_dims(img, axis=0)


def predict_tumor(image_path):
    IMAGE_SIZE = 128

    img = preprocess_image(image_path)

    predictions = model.predict(img, verbose=0)

    predicted_class_index = np.argmax(predictions)
    confidence_score = float(np.max(predictions))

    label = class_labels[predicted_class_index]

    if label == 'notumor':
        return "No Tumor", confidence_score
    elif label == 'pituitary':
        return "Pituitary", confidence_score
    else:
        return f"Tumor: {label}", confidence_score


# ========================
# Routes
# ========================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # Check file presence
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if not allowed_file(file.filename):
            return "Invalid file type"

        # Secure filename
        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Prediction
        result, confidence = predict_tumor(file_path)

        return render_template(
            'index.html',
            result=result,
            confidence=f"{confidence * 100:.2f}",
            file_path=f"/uploads/{filename}"
        )

    return render_template('index.html', result=None)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ========================
# Run App
# ========================
if __name__ == '__main__':
    app.run(debug=True)