from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Load model yang sudah disimpan
model = load_model("model_penyakit.h5")

# Ganti ini sesuai label penyakit kamu
label_encoder_classes = ['bacterial_pneumonia', 'covid19', 'normal', 'tuberculosis', 'viral_pneumonia']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def about():
    return render_template('home.html')

@app.route('/blog')
def about_page():
    return render_template('blog.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']

    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((64, 64))
        img_array = np.array(img)

        # === Validasi Brightness(gambar terlalu gelap/terang)===
        mean_brightness = np.mean(img_array)
        if mean_brightness < 30 or mean_brightness > 220:
            return render_template('index.html', prediction="Pencahayaan gambar ekstrim. Gambar gagal diproses(kemungkinan bukan x-ray).")

        #===kontras (ciri khas X-ray: ada kontras hitam-putih)===
        contrast = img_array.std()
        if contrast < 15:
            return render_template('index.html', prediction="Kontras gambar sangat rendah. Mungkin bukan X-ray.")

        # === Normalisasi & Prediksi ===
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 64, 64, 3)

        prediction = model.predict(img_array)[0]
        max_index = np.argmax(prediction)
        label = label_encoder_classes[max_index]
        confidence = prediction[max_index] * 100

        hasil = f"{label} ({confidence:.2f}%)"
        return render_template('index.html', prediction=hasil)

    except Exception as e:
        return render_template('index.html', prediction=f"Terjadi error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

