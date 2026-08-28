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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ── Load model + scaler at startup ───────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
model  = bundle["model"]
scaler = bundle["scaler"]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

# ── Feature extraction using scipy (no librosa needed) ───────────────────────
def extract_features(filepath):
    import wave, struct
    from scipy.fft import dct
    from scipy.signal import get_window

    with wave.open(filepath, 'r') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        n = min(n_frames, len(raw) // 2)
        samples = np.array(struct.unpack(f"{n}h", raw[:n*2]), dtype=float)

    if np.max(np.abs(samples)) > 0:
        samples = samples / np.max(np.abs(samples))

    samples = samples[:sr * 3]
    if len(samples) < sr * 3:
        samples = np.pad(samples, (0, sr * 3 - len(samples)))

    samples = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])

    n_mfcc = 40
    n_fft = 2048
    hop = 512
    n_mels = 128

    frames = []
    for i in range(0, len(samples) - n_fft, hop):
        frame = samples[i:i+n_fft] * get_window('hann', n_fft)
        frames.append(frame)
    frames = np.array(frames)

    power = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2

    fmin, fmax = 0, sr // 2
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    mel_energy = np.dot(power, fbank.T)
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
    log_mel = np.log(mel_energy)

    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    mfcc_mean = np.mean(mfcc, axis=0)      # 40
    mfcc_delta = np.mean(np.diff(mfcc, axis=0), axis=0)   # 40
    mfcc_delta2 = np.mean(np.diff(mfcc, n=2, axis=0), axis=0)  # 40

    zcr = np.array([np.mean(np.abs(np.diff(np.sign(samples))))/2])  # 1
    ste = np.array([np.mean(samples**2)])  # 1
    pitch = np.array([0.0, 0.0, 0.0])  # 3

    features = np.concatenate([mfcc_mean, mfcc_delta, mfcc_delta2, pitch, ste, zcr])  # 125
    return scaler.transform(features.reshape(1, -1))

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
        probs    = model.predict_proba(features)[0]
        classes  = model.classes_

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
