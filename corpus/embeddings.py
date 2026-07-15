"""
wraps InLegalBERT to turn Passage text into 768-dim vectors.

uses the [CLS] token's final hidden state as the passage embedding - the
standard trick for pulling a "sentence embedding" out of a BERT-family
model that wasn't specifically trained with a pooling objective.
"""

from asyncio import constants

import torch
from transformers import AutoTokenizer, AutoModel

from corpus.schemas import Passage
from config import constants
MODEL_NAME = constants.MODEL_NAME  # "nlpaueb/legal-bert-base-uncased"
BATCH_SIZE = constants.BATCH_SIZE  # same 6GB VRAM discipline as OCR/model inference - keep it small


class PassageEmbedder:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_passages(self, passages: list[Passage]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(passages), BATCH_SIZE):
            batch = passages[i:i + BATCH_SIZE]
            texts = [p.text for p in batch]

            inputs = self.tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            cls_vectors = outputs.last_hidden_state[:, 0, :]  # [CLS] token per sequence
            vectors.extend(cls_vectors.cpu().tolist())

        return vectors