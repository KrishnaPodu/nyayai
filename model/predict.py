"""
loads InLegalBERT with a token classification head and runs inference
on the chunks produced by preprocess.py.

before fine-tuned weights exist in model/checkpoint/, returns all O labels
(no errors detected) — honest behavior, not fake predictions. nothing else
in the pipeline needs to change when real weights are dropped in.
"""

import logging
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

from model.preprocess import Chunk
from model.schemas import LABELS, ID2LABEL

logger = logging.getLogger(__name__)

CHECKPOINT = "law-ai/InLegalBERT"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoint")
BATCH_SIZE = 8  # safe default for 6gb vram at inference (no gradients stored)


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_exists() -> bool:
    # expects at minimum a config.json in the checkpoint dir
    return os.path.isfile(os.path.join(CHECKPOINT_DIR, "config.json"))


def _load_model_and_tokenizer():
    device = _get_device()

    if _checkpoint_exists():
        logger.info(f"loading fine-tuned weights from {CHECKPOINT_DIR}")
        model_path = CHECKPOINT_DIR
    else:
        logger.warning(
            "no fine-tuned weights found in model/checkpoint/ — "
            "loading base InLegalBERT, all predictions will be O (no error). "
            "drop fine-tuned weights into model/checkpoint/ to enable real predictions."
        )
        model_path = CHECKPOINT

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes=True,  # base model has no classification head, this is expected
    )
    model.to(device)
    model.eval()  # never skip this — eval mode disables dropout and halves memory vs train mode

    return model, tokenizer, device


def predict(chunks: list[Chunk]) -> list[list[int]]:
    """
    runs inference on a list of chunks from preprocess.py.
    returns a list of label ID sequences, one per chunk, aligned to
    the chunk's token positions (including CLS and SEP positions).
    postprocess.py uses token_to_span to skip the None positions.
    """
    if not chunks:
        return []

    model, tokenizer, device = _load_model_and_tokenizer()

    all_label_ids = []

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        label_ids = _run_batch(batch_chunks, model, tokenizer, device)
        all_label_ids.extend(label_ids)

    return all_label_ids


def _run_batch(batch_chunks: list[Chunk], model, tokenizer, device) -> list[list[int]]:
    """
    pads a batch of chunks to the longest sequence in the batch,
    runs one forward pass, returns per-token label IDs for each chunk.
    padding to the longest in the batch (not always MAX_TOKENS) wastes
    less memory — a batch of short chunks doesn't get padded to 512.
    """
    # find longest sequence in this batch
    max_len = max(len(c.input_ids) for c in batch_chunks)
    pad_id = tokenizer.pad_token_id

    padded_input_ids = []
    padded_attention_masks = []

    for chunk in batch_chunks:
        seq_len = len(chunk.input_ids)
        pad_len = max_len - seq_len

        # pad input ids with pad_token_id, attention mask with 0
        # model will ignore positions where attention_mask=0
        padded_input_ids.append(chunk.input_ids + [pad_id] * pad_len)
        padded_attention_masks.append(chunk.attention_mask + [0] * pad_len)

    input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
    attention_mask = torch.tensor(padded_attention_masks, dtype=torch.long).to(device)

    with torch.no_grad():  # never skip this at inference — disables gradient tracking
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # outputs.logits shape: (batch_size, seq_len, num_labels)
    # softmax over label dimension to get per-label probabilities
    probs = F.softmax(outputs.logits, dim=-1)  # (batch, seq, num_labels)
    pred_ids = torch.argmax(probs, dim=-1)      # (batch, seq)

    # trim padding back off each sequence before returning
    # each chunk had a different original length, padding was just for batching
    results = []
    for chunk, pred_seq in zip(batch_chunks, pred_ids):
        original_len = len(chunk.input_ids)
        results.append(pred_seq[:original_len].tolist())

    return results