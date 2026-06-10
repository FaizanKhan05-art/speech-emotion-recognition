import librosa
import numpy as np
import noisereduce as nr

TARGET_SR   = 22050
SEGMENT_LEN = 2.5
HOP_LEN_S   = 1.0

def load_audio(file_path, sr=TARGET_SR):
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    return audio, sample_rate

def reduce_noise(audio, sr):
    reduced = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.75)
    return reduced

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val == 0:
        return audio
    return audio / max_val

def remove_silence(audio, sr, top_db=25):
    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return audio
    trimmed = np.concatenate([audio[s:e] for s, e in intervals])
    return trimmed

def segment_audio(audio, sr, seg_len=SEGMENT_LEN, hop=HOP_LEN_S):
    seg_samples  = int(seg_len * sr)
    hop_samples  = int(hop * sr)
    segments     = []
    start        = 0
    while start < len(audio):
        end     = start + seg_samples
        segment = audio[start:end]
        if len(segment) < seg_samples:
            segment = np.pad(segment, (0, seg_samples - len(segment)))
        segments.append(segment)
        start += hop_samples
    return segments

def preprocess_audio(file_path, sr=TARGET_SR):
    audio, sr = load_audio(file_path, sr)
    audio     = reduce_noise(audio, sr)
    audio     = normalize_audio(audio)
    audio     = remove_silence(audio, sr)
    segments  = segment_audio(audio, sr)
    return segments, sr
