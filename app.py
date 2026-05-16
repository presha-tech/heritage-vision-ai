# app.py

import os
import sqlite3
import urllib.request

from flask import Flask, render_template, request

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# UPLOAD FOLDER
# ─────────────────────────────────────────────────────────────

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ─────────────────────────────────────────────────────────────

MODEL_PATH = "best_resnet_model.h5"

MODEL_URL = "https://huggingface.co/presha-tech/heritage-vision-model/resolve/main/best_resnet_model.h5"

if not os.path.exists(MODEL_PATH):

    print("Downloading model from Hugging Face...")

    urllib.request.urlretrieve(
        MODEL_URL,
        MODEL_PATH
    )

    print("Model downloaded successfully!")

# ─────────────────────────────────────────────────────────────
# LAZY LOAD MODEL
# ─────────────────────────────────────────────────────────────

_model = None

def get_model():

    global _model

    if _model is None:

        from tensorflow.keras.models import load_model

        print("Loading TensorFlow model...")

        _model = load_model(MODEL_PATH)

        print("Model loaded successfully!")

    return _model

# ─────────────────────────────────────────────────────────────
# CLASS LABELS
# ─────────────────────────────────────────────────────────────

CLASS_NAMES = [

    "Charminar",

    "Gateway of India",

    "Qutub Minar",

    "Sun Temple Konark",

    "Taj Mahal",
]

# ─────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────

def predict_monument(filepath):

    import numpy as np

    from tensorflow.keras.preprocessing import image

    from tensorflow.keras.applications.resnet50 import preprocess_input

    model = get_model()

    img = image.load_img(
        filepath,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = preprocess_input(
        img_array[None, ...]
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = int(
        predictions.argmax()
    )

    confidence = float(
        predictions.max()
    ) * 100

    monument = CLASS_NAMES[predicted_index]

    return monument, confidence

# ─────────────────────────────────────────────────────────────
# HOME ROUTE
# ─────────────────────────────────────────────────────────────

@app.route("/")
def home():

    return render_template("index.html")

# ─────────────────────────────────────────────────────────────
# PREDICT ROUTE
# ─────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():

    try:

        if "image" not in request.files:

            return render_template(
                "index.html",
                error="No image uploaded."
            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(
                "index.html",
                error="No image selected."
            )

        # Secure filename

        from werkzeug.utils import secure_filename

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        # Save uploaded image

        file.save(filepath)

        # Predict monument

        monument, confidence = predict_monument(
            filepath
        )

        # ─────────────────────────────────────────────────────
        # FETCH HISTORICAL INFORMATION
        # ─────────────────────────────────────────────────────

        db_path = os.path.join(
            os.path.dirname(__file__),
            "monuments.db"
        )

        conn = sqlite3.connect(db_path)

        cursor = conn.cursor()

        cursor.execute(
            """

            SELECT
                history,
                dynasty,
                construction_period,
                architecture,
                unesco_status,
                tourism_facts

            FROM monuments

            WHERE name = ?

            """,
            (monument,)
        )

        row = cursor.fetchone()

        conn.close()

        NOT_AVAILABLE = "Information not available."

        if row:

            history = row[0]
            dynasty = row[1]
            construction_period = row[2]
            architecture = row[3]
            unesco_status = row[4]
            tourism_facts = row[5]

        else:

            history = NOT_AVAILABLE
            dynasty = NOT_AVAILABLE
            construction_period = NOT_AVAILABLE
            architecture = NOT_AVAILABLE
            unesco_status = NOT_AVAILABLE
            tourism_facts = NOT_AVAILABLE

        # ─────────────────────────────────────────────────────
        # RETURN RESULT PAGE
        # ─────────────────────────────────────────────────────

        return render_template(

            "result.html",

            monument=monument,

            confidence=f"{confidence:.2f}",

            image_path=filepath,

            history=history,

            dynasty=dynasty,

            construction_period=construction_period,

            architecture=architecture,

            unesco_status=unesco_status,

            tourism_facts=tourism_facts
        )

    except Exception as e:

        return f"ERROR: {str(e)}"

# ─────────────────────────────────────────────────────────────
# RUN APP
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )