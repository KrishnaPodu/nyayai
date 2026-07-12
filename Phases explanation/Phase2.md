# phase 2 — model / error detection (`feature/model`)

## goal

take the `list[LineSpan]` that phase 1 produces and return a
`list[ErrorSpan]` — each span carrying the flagged text, error type,
bounding box, and confidence score.

single public API:
```python
from model.pipeline import analyze

errors = analyze(spans)  # spans is list[LineSpan] from ocr.pipeline.extract()
# returns list[ErrorSpan]
```

---

## honest picture before  start

**InLegalBERT has no error-detection head out of the box.**
`law-ai/InLegalBERT` is pretrained on Indian legal text (good — it
understands legal vocabulary, citations, IPC references). but its
existing task heads are for judgment outcome prediction and document
segmentation — not "is this word misspelled" or "is this section number
wrong." that head doesn't exist yet.

so phase 2 builds the full inference scaffold with the correct
architecture. before fine-tuned weights exist, `predict.py` returns all
`O` (no error) labels — honest, not fake. the moment you drop fine-tuned
weights into `model/checkpoint/`, real predictions start flowing through
without changing a single line of code elsewhere.

fine-tuning itself (generating synthetic data + training) is a separate
branch (`feature/finetune`) that runs in parallel and feeds weights back
into this scaffold.

**exception: citation checking doesn't need fine-tuning at all.**
`citation_checker.py` is pure retrieval — regex extracts citation spans,
Qdrant checks them against the IPC/BNS/Constitution corpus. this works
and produces real results right now, independently of the model.

---

## files to build (in order)

```
model/
├── __init__.py
├── schemas.py           DONE - ErrorSpan dataclass, BIO label definitions
├── preprocess.py        groups LineSpans into model-ready chunks
├── predict.py           loads InLegalBERT, runs token classification
├── postprocess.py       merges BIO token labels into ErrorSpans
├── citation_checker.py  checks citation spans against Qdrant
└── pipeline.py          analyze(spans) -> list[ErrorSpan]
```

---

## what each file does

### `schemas.py` — done
defines `ErrorSpan` and the BIO label scheme used across all files:
- `O` — correct
- `B-SPELL / I-SPELL` — spelling error
- `B-GRAM / I-GRAM` — grammar error
- `B-CITE / I-CITE` — wrong IPC/BNS citation

`LABEL2ID` and `ID2LABEL` live here as the single source of truth.
every other file imports from here — no label mapping defined anywhere else.

---

### `preprocess.py`
**what it does:**
InLegalBERT is a BERT model with a 512 subword token limit. a long legal
document has hundreds of lines. preprocess groups `LineSpan`s into chunks
that fit within that limit, and critically keeps a mapping from each
subword token back to its source `LineSpan` — so when predict.py says
"token 47 is B-CITE", postprocess can look up which LineSpan that token
came from and get its bbox.

**inputs:** `list[LineSpan]`
**outputs:** list of `Chunk` objects, each containing:
- tokenized input ids, attention mask (ready to pass to the model)
- a `token_to_span` mapping: `list[int | None]` where index is token
  position and value is the index into the original spans list
  (`None` for [CLS], [SEP], and subword continuations)

**industry practice — subword alignment:**
BERT doesn't tokenize by whitespace. "Section" might become ["Section"]
but "IPC" might become ["IP", "##C"]. only the first subword of each
word gets a real label — continuations get `-100` so they're ignored
in loss computation during training. this is the standard HuggingFace
convention and `word_ids()` from the tokenizer is how you implement it.

**industry practice — stride/overlap for long documents:**
rather than hard-cutting at 512 tokens (losing context at boundaries),
use a sliding window with overlap — e.g. chunks of 512 tokens with 128
token stride. this way a citation that spans a chunk boundary still gets
seen in full context by at least one chunk.

---

### `predict.py`
**what it does:**
loads `law-ai/InLegalBERT` with a token classification head
(`AutoModelForTokenClassification`), runs inference on preprocessed
chunks, returns raw label IDs per token.

**inputs:** list of `Chunk` objects from preprocess
**outputs:** list of label ID sequences, one per chunk

**before fine-tuned weights exist:**
looks for weights at `model/checkpoint/`. if not found, logs a warning
and returns all `O` labels (no errors detected). this is the correct
honest behavior — not an exception, not fake predictions.

**after fine-tuned weights exist:**
drop weights into `model/checkpoint/` and predictions become real.
nothing else changes.

**industry practice — inference only:**
always call `model.eval()` and wrap inference in `torch.no_grad()`.
skips gradient computation entirely, roughly halves memory usage, speeds
up inference. not optional, not "nice to have" — forgetting this is a
common bug that silently wastes vram.

**industry practice — device handling:**
check `torch.cuda.is_available()` and move model + inputs to GPU if
available, CPU otherwise. never hardcode `"cuda"` — makes the code
break on machines without a GPU.

**industry practice — batching:**
don't run one chunk at a time through the model. batch multiple chunks
per forward pass up to the vram limit. for the 4050 6gb card, batch
size 8 is a reasonable default for inference (larger than training
since we don't need to store gradients).

---

### `postprocess.py`
**what it does:**
takes raw per-token label IDs from predict.py and the chunk objects from
preprocess.py and reconstructs actual `ErrorSpan`s with real bboxes.

steps:
1. for each token with a `B-` label, start a new span
2. extend it with following `I-` tokens of the same type
3. collect the source `LineSpan` bboxes for all tokens in the span
4. merge bboxes into one tight box covering the full error span
5. construct `ErrorSpan` with merged bbox, error type, confidence

**inputs:** chunks + raw label sequences from predict
**outputs:** `list[ErrorSpan]`

**industry practice — confidence from logits:**
don't just take argmax of logits for the label. also run softmax and
store the probability of the predicted class as `confidence` on the
`ErrorSpan`. gives the frontend something to threshold on — e.g. only
show errors with confidence > 0.7.

**industry practice — bbox merging:**
when a span covers multiple tokens that might come from different lines
(rare but possible at chunk boundaries), take:
- `x0 = min(all x0s)`
- `x1 = max(all x1s)`
- `y0 = min(all y0s)`
- `y1 = max(all y1s)`

this gives a tight bounding box that covers all tokens in the span.

---

### `citation_checker.py`
**what it does:**
independently of the ML model, extracts citation patterns from spans
using regex and checks them against the IPC/BNS/Constitution corpus
stored in Qdrant. flags citations that don't match any known valid
section as `B-CITE` errors.

patterns it looks for:
- `Section \d+ IPC`
- `Section \d+ BNS`
- `u/s \d+`
- `Article \d+ of the Constitution`

for each match, queries Qdrant for that section number. if no match or
section is repealed post-BNS, flags it as a citation error.

**inputs:** `list[LineSpan]`
**outputs:** `list[ErrorSpan]` (citation errors only)

**industry practice — don't duplicate ML model's work:**
citation checker only runs on spans the ML model didn't already flag as
`CITE` errors. in `pipeline.py`, ML errors come first, citation checker
fills in any citations the model missed. no double-flagging.

**industry practice — Qdrant connection handling:**
wrap Qdrant client calls in try/except. if Qdrant is down or the
collection doesn't exist yet (corpus not ingested), log a warning and
return empty list — don't crash the whole pipeline because the vector
db isn't running.

---

### `pipeline.py`
**what it does:**
the single entry point for phase 2. calls preprocess → predict →
postprocess, then citation checker, deduplicates overlapping spans,
and returns the final sorted list of `ErrorSpan`s.

```python
from model.pipeline import analyze
errors = analyze(spans)  # list[LineSpan] in, list[ErrorSpan] out
```

**industry practice — deduplication:**
ML model and citation checker might flag the same span. before returning,
deduplicate by checking bbox overlap. keep the one with higher confidence.

**industry practice — sort output:**
return errors sorted by `(page_no, y0, x0)` — reading order. makes the
frontend's job trivial and makes debugging output human-readable.

---

## done when

```python
from ocr.pipeline import extract
from model.pipeline import analyze

spans = extract("some_fir.pdf")
errors = analyze(spans)

for e in errors:
    print(e.page_no, e.error_type, e.text, e.bbox, e.confidence)
```

runs without crashing. returns empty list if no fine-tuned weights exist.
returns real errors once weights land in `model/checkpoint/`.

---

## what this phase does NOT do

- train the model (that's `feature/finetune`)
- ingest the IPC/BNS corpus into Qdrant (that's `feature/corpus`)
- generate suggested corrections (future work, separate model)
- serve results over HTTP (that's `feature/api`)