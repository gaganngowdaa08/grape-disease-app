from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB limit
app.config['UPLOAD_FOLDER'] = 'static'

# Load model
model = None
def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('model/grape_model.h5')
    return model

# Disease information
disease_classes = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]
disease_classes_kn = ["ಕಪ್ಪು ಕೀಟ", "ಎಸ್ಕಾ", "ಆರೋಗ್ಯಕರ", "ಬ್ಲೈಟ್ ಎಲೆ"]

disease_tips = {
    "Black Rot": "Remove infected leaves and apply fungicide early in the season.",
    "ESCA": "Prune infected vines and avoid water stress. No chemical cure available.",
    "Healthy": "Your plant looks good! Keep monitoring and maintain hygiene.",
    "Leaf Blight": "Use protective sprays and remove affected leaves promptly."
}

disease_tips_kn = {
    "Black Rot": "ಸೋಂಕು ಹರಡಿದ ಎಲೆಗಳನ್ನು ತೆಗೆದುಹಾಕಿ ಮತ್ತು ಸೀಜನ್ ಆರಂಭದಲ್ಲಿ ಫಂಗಸೈಡ್ ಅನ್ನು ಅನ್ವಯಿಸಿ.",
    "ESCA": "ಸೋಂಕು ಬಿದಿರುಗಳನ್ನು ಕತ್ತರಿಸಿ ಮತ್ತು ನೀರಿನ ಒತ್ತಡವನ್ನು ತಪ್ಪಿಸಿ. ರಾಸಾಯನಿಕ ಚಿಕಿತ್ಸೆ ಲಭ್ಯವಿಲ್ಲ.",
    "Healthy": "ನಿಮ್ಮ ದ್ರಾಕ್ಷಿ ಬಳ್ಳಿ ಆರೋಗ್ಯಕರವಾಗಿ ಕಾಣುತ್ತದೆ. ನಿರಂತರವಾಗಿ ಗಮನಿಸಿ ಮತ್ತು ಸ್ವಚ್ಛತೆಯನ್ನು ಕಾಪಾಡಿ.",
    "Leaf Blight": "ರಕ್ಷಕ ಸಿಂಪಡಣೆಗಳನ್ನು ಬಳಸಿ ಮತ್ತು ಪರಿಣಾಮ ಬೀರಿದ ಎಲೆಗಳನ್ನು ತಕ್ಷಣ ತೆಗೆದುಹಾಕಿ."
}

@app.route('/')
def home():
    # Cleanup old uploaded images
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        if f.startswith("uploaded_"):
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
            except Exception as e:
                print(f"Error deleting file {f}: {e}")
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save uploaded file
        filename = f"uploaded_{np.random.randint(10000)}.jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(image_path)

        # Preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        model = load_model()
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_label = disease_classes[predicted_index]
        confidence = float(np.max(prediction)) * 100

        # Language support
        lang = request.form.get('language', 'en')
        if lang == 'kn':
            predicted_label = disease_classes_kn[predicted_index]
            tip = disease_tips_kn.get(disease_classes[predicted_index], "No tips available")
        else:
            tip = disease_tips.get(predicted_label, "No tips available")

        return jsonify({
            'status': 'success',
            'prediction': predicted_label,
            'confidence': round(confidence, 2),
            'treatment': tip,
            'image_url': image_path
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form.get('message', '').lower()
    lang = request.form.get('language', 'en')

    # Greetings
    if any(greet in user_input for greet in ['hi', 'hello', 'hey', 'ನಮಸ್ಕಾರ']):
        response = "Hello! Ask me about grape diseases." if lang == 'en' else "ನಮಸ್ಕಾರ! ದ್ರಾಕ್ಷಿ ಕಾಯಿಲೆಗಳ ಬಗ್ಗೆ ಕೇಳಿ."
        return jsonify({'response': response})

    # Disease queries
    for i, disease in enumerate(disease_classes):
        if disease.lower() in user_input or disease_classes_kn[i].lower() in user_input:
            tip = disease_tips[disease] if lang == 'en' else disease_tips_kn[disease]
            response = f"{disease if lang == 'en' else disease_classes_kn[i]}: {tip}"
            return jsonify({'response': response})

    # Default response
    default = ("Ask about grape diseases like Black Rot, ESCA, or Leaf Blight." if lang == 'en' 
               else "ಕಪ್ಪು ಕೀಟ, ಎಸ್ಕಾ, ಅಥವಾ ಬ್ಲೈಟ್ ಎಲೆಗಳಂತಹ ದ್ರಾಕ್ಷಿ ಕಾಯಿಲೆಗಳ ಬಗ್ಗೆ ಕೇಳಿ.")
    return jsonify({'response': default})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
