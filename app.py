# app.py
from flask import Flask, request
import os
from datetime import datetime

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    label = request.form.get('label')
    image = request.files.get('image')

    if not label or not image:
        return "Missing label or image", 400

    save_dir = os.path.join("training_data", label)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(save_dir, f"{timestamp}.jpg")
    image.save(path)

    return "Uploaded successfully", 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
