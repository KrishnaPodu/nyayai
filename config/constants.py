# -------------------------
# InLegalBERT
# -------------------------

MAX_TOKENS = 512

CHUNK_STRIDE = 128

INFERENCE_BATCH_SIZE = 8

# -------------------------
# Rendering
# -------------------------

ERROR_COLORS = {
    "spelling": "#FFD700",
    "grammar": "#FFA500",
    "citation": "#FF4444",
    "entity": "#00BFFF",
}

# Embeddings constants for corpus/embeddings.py
MODEL_NAME = "law-ai/InLegalBERT"
BATCH_SIZE = 16

# acts for parsing and chunking
ACTS = ["ipc", "bns", "bnss", "cpc", "constitution"]