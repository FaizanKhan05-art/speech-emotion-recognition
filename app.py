import os
import uuid
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# ── Model & label config ─────────────────────────────────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "ser_model.pkl")
UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXT    = {"wav"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB cap

# ── Load model once at startup ───────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ── Feature extraction ───────────────────────────────────────────────────────
def extract_features(filepath):
    import librosa
    audio, sr = librosa.load(filepath, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", emotions=EMOTION_LABELS)

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request"}), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .wav files are supported"}), 415

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        features = extract_features(filepath)
        probs = model.predict_proba(features)[0]
        classes = model.classes_

        best_idx   = int(np.argmax(probs))
        predicted  = classes[best_idx]
        confidence = float(probs[best_idx])
        prob_dict  = {lbl: round(float(p), 4) for lbl, p in zip(classes, probs)}

        return jsonify({
            "predicted_emotion": predicted,
            "confidence":        round(confidence * 100, 2),
            "probabilities":     prob_dict,
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
