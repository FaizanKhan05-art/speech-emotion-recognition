import librosa
import numpy as np

N_MFCC  = 40
N_MELS  = 128
N_FFT   = 2048
HOP     = 512

def extract_mfcc(audio, sr, n_mfcc=N_MFCC):
    mfcc    = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc,
                                   n_fft=N_FFT, hop_length=HOP)
    delta   = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)
    stacked = np.concatenate([mfcc, delta, delta2], axis=0)
    return np.mean(stacked.T, axis=0)

def extract_spectrogram(audio, sr, n_mels=N_MELS, fixed_len=128):
    mel     = librosa.feature.melspectrogram(y=audio, sr=sr,
                                             n_fft=N_FFT,
                                             hop_length=HOP,
                                             n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape[1] < fixed_len:
        pad = fixed_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)))
    else:
        log_mel = log_mel[:, :fixed_len]
    return log_mel[..., np.newaxis]

def extract_pitch(audio, sr):
    f0, _, _ = librosa.pyin(audio,
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'),
                            sr=sr)
    f0_voiced = f0[~np.isnan(f0)]
    if len(f0_voiced) == 0:
        return np.zeros(3)
    return np.array([np.mean(f0_voiced),
                     np.std(f0_voiced),
                     np.max(f0_voiced)])

def extract_all_features(audio, sr):
    mfcc_feat  = extract_mfcc(audio, sr)
    pitch_feat = extract_pitch(audio, sr)
    ste        = np.array([np.mean(audio ** 2)])
    zcr        = np.array([np.mean(
                    librosa.feature.zero_crossing_rate(audio))])
    return np.concatenate([mfcc_feat, pitch_feat, ste, zcr])
