import os
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# ── Model & label config ────────────────────────────────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "best_lstm_model.h5")
UPLOAD_FOLDER  = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXT    = {"wav"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB cap

# ── Lazy-load model (avoids TF import cost at startup if not needed) ─────────
_model = None

def get_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model  # type: ignore
        _model = load_model(MODEL_PATH)
    return _model

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ── Preprocessing & feature extraction (mirrors your pipeline) ──────────────
def predict_from_file(filepath):
    """
    Run the full pipeline:
    preprocess → extract features per segment → aggregate → model inference.
    Returns (predicted_label, confidence, per-class probabilities dict).
    """
    from preprocessing import preprocess_audio
    from features import extract_all_features

    segments, sr = preprocess_audio(filepath)

    # Extract (125,) feature vector for every segment
    feature_vectors = np.array([extract_all_features(seg, sr) for seg in segments])

    # BiLSTM expects (batch, timesteps, features).
    # Each segment is one timestep; treat all segments as the sequence.
    X = feature_vectors[np.newaxis, :, :]          # (1, T, 125)

    model = get_model()
    probs = model.predict(X, verbose=0)[0]          # (num_classes,)

    # Align to label list length defensively
    n = min(len(probs), len(EMOTION_LABELS))
    probs = probs[:n]
    labels = EMOTION_LABELS[:n]

    best_idx    = int(np.argmax(probs))
    predicted   = labels[best_idx]
    confidence  = float(probs[best_idx])
    prob_dict   = {lbl: round(float(p), 4) for lbl, p in zip(labels, probs)}

    return predicted, confidence, prob_dict

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", emotions=EMOTION_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request (field name: 'audio')"}), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only .wav files are supported"}), 415

    # Save to a temp path to avoid name collisions
    filename   = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath   = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        label, confidence, probs = predict_from_file(filepath)
        return jsonify({
            "predicted_emotion": label,
            "confidence":        round(confidence * 100, 2),
            "probabilities":     probs,
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        # Clean up uploaded file after prediction
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
