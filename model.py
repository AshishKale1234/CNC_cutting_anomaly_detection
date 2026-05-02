# ── model.py ─────────────────────────────────────────────────────────────────
# Mel spectrogram extraction + TFLite inference
# Parameters match training notebook exactly

import numpy as np
import librosa
import tensorflow as tf
from config import (
    SAMP_RATE, N_FFT, HOP_LENGTH, N_MELS,
    WINDOW_SEC, WINDOW_SAMP, CHATTER_THRESHOLD,
    TFLITE_MODEL_PATH
)


def extract_mel_feature(audio_window: np.ndarray, sr: int = SAMP_RATE) -> np.ndarray:
    """
    Compute Mel spectrogram feature from a 1-second audio window.
    Exactly matches Lab 10 / training notebook feature extraction.
    Returns shape: (N_MELS, time_frames) = (128, 126)
    """
    M = librosa.feature.melspectrogram(
        y          = audio_window,
        sr         = sr,
        n_fft      = N_FFT,
        hop_length = HOP_LENGTH,
        win_length = N_FFT,
        window     = 'hann',
        n_mels     = N_MELS
    )
    feature = 2 * abs(M) / N_FFT   # same normalization as Lab 10
    return feature.astype(np.float32)


class ChatterModel:
    """
    TFLite inference wrapper.
    Loads model once, reuses interpreter for all predictions.
    """

    def __init__(self, model_path: str = TFLITE_MODEL_PATH):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"[Model] Loaded: {model_path}")
        print(f"[Model] Input shape: {self.input_details[0]['shape']}")

    def predict_window(self, audio_window: np.ndarray) -> float:
        """
        Run inference on a single 1-second audio window.
        Returns chatter probability (0.0 to 1.0).
        """
        feat = extract_mel_feature(audio_window)
        feat = np.expand_dims(feat, axis=0)   # (1, 128, 126)

        self.interpreter.set_tensor(self.input_details[0]['index'], feat)
        self.interpreter.invoke()
        prob = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        return float(prob)

    def predict_audio(self, audio: np.ndarray, sr: int = SAMP_RATE):
        """
        Run inference on a full audio array.
        Chops into 1s windows, predicts each, returns probs + flags + index.

        Returns:
            probs         : np.array of chatter probabilities per window
            chatter_flags : np.array of booleans (True = chatter)
            chatter_index : float (0-100) percentage of chatter windows
            n_windows     : int
        """
        window_samples = int(sr * WINDOW_SEC)
        n_windows      = len(audio) // window_samples
        probs          = []

        for i in range(n_windows):
            chunk = audio[i * window_samples : (i + 1) * window_samples]
            prob  = self.predict_window(chunk)
            probs.append(prob)

        probs         = np.array(probs)
        chatter_flags = probs >= CHATTER_THRESHOLD
        chatter_index = float(np.mean(chatter_flags) * 100)

        return probs, chatter_flags, chatter_index, n_windows
