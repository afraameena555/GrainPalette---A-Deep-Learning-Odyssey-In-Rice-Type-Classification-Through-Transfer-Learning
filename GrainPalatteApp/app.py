import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import send_from_directory


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model/rice_classification_model.keras')
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/details', methods=['POST'])
def details():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    image_url = '/uploads/' + file.filename
    return render_template('details.html', image_url=image_url)

@app.route('/predict', methods=['POST'])
def predict():
    from tensorflow.keras.preprocessing import image
    import numpy as np
    import os

    # Get relative path to uploaded image
    filepath = request.form['filepath']  # e.g., /uploads/filename.jpg
    filename = os.path.basename(filepath)  # Just the filename
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Load and preprocess image
    img = image.load_img(full_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Predict
    pred = model.predict(x)[0]
    label = class_names[np.argmax(pred)]
    confidence = round(100 * np.max(pred), 2)

    # Return to results page
    image_url = '/uploads/' + filename
    return render_template('results.html', label=label, confidence=confidence, image_url=image_url)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
