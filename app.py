# app.py

from flask import Flask, render_template, request
import os
import subprocess
import sqlite3

app = Flask(__name__)

# =========================
# UPLOAD FOLDER
# =========================

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder automatically
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():
    return render_template("index.html")

# =========================
# PREDICTION
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    # =========================
    # CHECK IMAGE
    # =========================

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No selected file"

    # =========================
    # SAVE IMAGE
    # =========================

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # =========================
    # RUN AI MODEL
    # =========================

    result = subprocess.check_output(
        ["python", "predict.py", filepath]
    ).decode("utf-8")

    # Clean TensorFlow logs if present
    result_lines = result.strip().split("\n")

    final_result = result_lines[-1]

    monument, confidence = final_result.split("|")

    # =========================
    # FETCH MONUMENT INFO
    # =========================

    conn = sqlite3.connect("monuments.db")

    cursor = conn.cursor()

    cursor.execute("""

    SELECT
        history,
        dynasty,
        construction_period,
        architecture,
        unesco_status,
        tourism_facts

    FROM monuments

    WHERE name = ?

    """, (monument,))

    data = cursor.fetchone()

    conn.close()

    # =========================
    # HANDLE MISSING DATA
    # =========================

    if data:

        history = data[0]
        dynasty = data[1]
        construction_period = data[2]
        architecture = data[3]
        unesco_status = data[4]
        tourism_facts = data[5]

    else:

        history = "Information not available."
        dynasty = "Information not available."
        construction_period = "Information not available."
        architecture = "Information not available."
        unesco_status = "Information not available."
        tourism_facts = "Information not available."

    # =========================
    # RETURN RESULT PAGE
    # =========================

    return render_template(

        "result.html",

        monument=monument,

        confidence=confidence,

        image_path=filepath,

        history=history,

        dynasty=dynasty,

        construction_period=construction_period,

        architecture=architecture,

        unesco_status=unesco_status,

        tourism_facts=tourism_facts
    )

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(debug=True)