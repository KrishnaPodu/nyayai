

## how the project is split

doing this in phases, one git branch per phase. plan below.

---

### phase 1 - ocr (`feature/ocr`)

goal: pdf in -> list of WordToken out. works for both normal pdfs and scanned ones.

order to build:
1. `ocr/tokens.py` - WordToken dataclass (done)
2. `ocr/native_extractor.py` - NativeExtractor, pulls text from pdfs that already have text layer
3. `ocr/surya_extractor.py` - SuryaExtractor, runs surya OCR for scanned pdfs (gpu)
4. `ocr/router.py` - decides per page if it needs OCR or not, sends to right extractor
5. `ocr/pipeline.py` - combines everything into one extract(pdf_path) function

done when: any pdf in -> flat list of WordTokens with correct bbox + page no.

---

### phase 2 - model (`feature/model`)

goal: tokens in -> error spans out (spelling/grammar/wrong citation)

order to build:
1. `model/preprocess.py` - groups WordTokens into lines/sentences for the model, keeps mapping back to bbox
2. `model/predict.py` - loads law-ai/InLegalBERT, runs inference, gives raw per token labels
3. `model/postprocess.py` - merges token labels into proper spans (so "Section 302 IPC" is 1 span not 4)
4. `model/citation_checker.py` - checks flagged citations against qdrant (is this section even real / still active)
5. `model/pipeline.py` - analyze(tokens) -> list of ErrorSpan, single entry point

depends on phase 1 (needs WordToken).

done when: feed it tokens from phase 1, get back error spans with bbox + type + suggested fix.

---

### phase 3 - corpus (`feature/corpus`)

goal: get IPC/BNS/constitution text into qdrant so citation_checker has something
to check against. can be done in parallel with phase 2, they only meet at
citation_checker.py

order to build:
1. `scripts/ingest_corpus.py` - parse raw IPC/BNS/constitution docs into clean sections
2. `scripts/embed_and_upload.py` - embed each section + upload to qdrant with metadata (section no, act, active/repealed)
3. `scripts/verify_corpus.py` - quick script to test a few queries actually return right section

done when: citation_checker.py can query qdrant and get correct section + status back.

---

### phase 4 - api (`feature/api`)

goal: wrap phase 1+2+3 as an async job so frontend isnt stuck waiting

order to build:
1. `api/main.py` - fastapi app, /upload /status/{job_id} /result/{job_id}
2. `api/celery_app.py` - celery + redis config
3. `api/tasks.py` - celery task that runs ocr pipeline -> model pipeline
4. `api/schemas.py` - pydantic models for request/response

depends on phase 2 + phase 3 being mostly done.

done when: can POST a pdf, poll status, get json with error spans once job is done.

---

### phase 5 - frontend (`feature/frontend`)

goal: show pdf with color coded highlights over the errors

order to build:
1. `frontend/src/PdfCanvas.jsx` - renders pdf onto canvas using pdf.js
2. `frontend/src/HighlightOverlay.jsx` - draws the colored boxes using bbox + error type from api
3. `frontend/src/api.js` - upload / poll / fetch result calls
4. `frontend/src/App.jsx` - wires all of it together

depends on phase 4 (needs a working api to hit).

done when: upload pdf in browser, see it render, see highlights show up after backend finishes.

---

## branch merge order

```
main
 -> feature/ocr        (merge first, nothing works without WordToken)
 -> feature/corpus      (parallel with model)
 -> feature/model        (after ocr merged)
 -> feature/api            (after model + corpus merged)
 -> feature/frontend         (last, needs live api)
```

## stack

- OCR: surya-ocr 0.9.3 (pinned, 0.20.0 needs vllm which we dont want)
- NLP: law-ai/InLegalBERT (token classification)
- torch 2.4.1+ with cu128 wheels (for cuda 13.2)
- transformers >=4.56.1,<6.0.0
- vector db: qdrant
- backend: fastapi + celery + redis
- frontend: react + pdf.js
- pkg manager: uv