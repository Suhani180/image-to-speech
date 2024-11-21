from flask import Flask, render_template, request
from test import generate_caption
import pyttsx3
import os

# Flask app initialization
app = Flask(__name__)
UPLOAD_FOLDER = './data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route: Home
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Upload image
        image = request.files["image"]
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)

        # Generate caption
        caption = generate_caption(image_path)

        # Convert to speech
        engine = pyttsx3.init()
        engine.say(caption)
        engine.runAndWait()

        return render_template("index.html", caption=caption, image=image.filename)
    return render_template("index.html", caption=None)

if __name__ == "__main__":
    app.run(debug=True)
