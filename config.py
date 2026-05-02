# ── config.py ────────────────────────────────────────────────────────────────
# All constants matching the trained model parameters exactly

# ── AUDIO / MODEL PARAMS (must match training notebook exactly) ──────────────
SAMP_RATE         = 16000       # Maijker sensor sample rate
N_FFT             = 512         # FFT points
HOP_LENGTH        = N_FFT // 4  # 128 samples
N_MELS            = 128         # Mel filter banks
WINDOW_SEC        = 1.0         # seconds per inference window
WINDOW_SAMP       = int(SAMP_RATE * WINDOW_SEC)  # 16000 samples
ANOMALY_THRESHOLD = 0.85        # tuned classification threshold

# ── BACKWARDS COMPATIBILITY ──────────────────────────────────────────────────
CHATTER_THRESHOLD = ANOMALY_THRESHOLD   # legacy alias for model.py

# ── MODEL PATH ───────────────────────────────────────────────────────────────
TFLITE_MODEL_PATH = "models/20260414_103802_anomaly_CNN_model.tflite"

# ── MAIJKER / SCRAPER ────────────────────────────────────────────────────────
EMAIL      = " " # Place your email that was used to create an account 
PASSWORD   = " " # Password
SENSOR_URL = "https://app.maijker.com/metrics/76587da3-cd8a-449a-bb58-47023f5da2a0"
LOGIN_URL  = (
    "https://maijker-cloud-identity-b3augeeyfcfbgvek.eastus-01.azurewebsites.net"
    "/Identity/Account/Login"
    "?returnUrl=https://app.maijker.com/handle-login&theme=light"
)

# ── LIVE MODE ────────────────────────────────────────────────────────────────
LIVE_POLL_INTERVAL_SEC = 30
LIVE_LOOKBACK_MIN      = 2

# ── CUTTING ANOMALY INDEX ZONES ──────────────────────────────────────────────
ZONE_STABLE   = 20   # below this → STABLE
ZONE_ABNORMAL = 50   # below this → ABNORMAL, above → CRITICAL

# Backwards compat alias
ZONE_MODERATE = ZONE_ABNORMAL