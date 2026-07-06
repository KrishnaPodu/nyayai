# nyayai

an AI tool that proofreads indian legal documents (FIRs, contracts, court notices etc) and highlights spelling, grammar and semantic errors.

built this as a personal project to learn NLP. still very much work in progress.

---

## what it does

upload a PDF → get back the same PDF with colored highlights showing:
- 🟡 spelling mistakes
- 🟠 grammar errors  
- 🔴 wrong IPC/BNS section references

---

## how it works (roughly)

```
PDF
 ↓
extract text page by page
  - if page has text → pymupdf (fast)
  - if page is scanned → surya OCR (slow but works)
 ↓
run InLegalBERT on the text
  - fine-tuned on Indian legal sentences
  - classifies each word as: correct / spelling error / grammar error / wrong citation
 ↓
for wrong citations → check against IPC/BNS/Constitution database (qdrant)
 ↓
draw colored boxes on the original PDF
 ↓
show in browser with React
```

---

## folder structure

this is what actually exists on disk right now, not the final planned structure.
more files get added as each phase progresses (see phase plan doc for order).

```
NyayAI/
│
├── ocr/
│   ├── __init__.py
│   ├── tokens.py            WordToken dataclass, shared by everything below
│   └── native_extractor.py  NativeExtractor - pulls text from pdfs that already have a text layer
│                            (surya_extractor.py and router.py not built yet)
│
├── model/                   empty for now, InLegalBERT pipeline goes here next
│
├── test_deps.py             checks pinned versions actually installed correctly
├── test_gpu.py              checks torch can see the gpu before doing anything heavy
└── README.md
```

planned but not built yet (coming up in order):

```
├── ocr/
│   ├── surya_extractor.py   SuryaExtractor for scanned pages
│   └── pipeline.py          combines native + surya behind one extract() call
│
├── model/
│   ├── checkpoint/          trained model goes here (not in git, too big)
│   └── detector.py          runs InLegalBERT inference on extracted words
│
├── api/
│   └── main.py              FastAPI server - handles uploads and job status
│
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── UploadPage.jsx
│       └── ResultPage.jsx   shows the annotated PDF with highlights
│
├── scripts/
│   ├── generate_data.py     makes training data from IL-TUR dataset
│   └── train.py             fine-tunes InLegalBERT
│
├── data/
│   ├── raw_pdfs/            uploaded pdfs (temp storage)
│   └── training_data/       train.jsonl, val.jsonl, test.jsonl
│
├── tests/
│   └── test_ocr.py
│
├── config.py                all settings in one place
├── main.py                  quick cli test script
├── requirements.txt
├── .env.example
└── docker-compose.yml       runs qdrant + redis (need these running)
```

---

## setup

**you need:**
- python 3.10
- NVIDIA GPU with CUDA 12.4 (i have RTX 4050 6GB)
- docker (for qdrant and redis)
- node 20+ (for frontend)

**install:**

```bash
# clone and go in
git clone <repo>
cd NyayAI

# create virtualenv
uv venv
source .venv/bin/activate

# pytorch FIRST (important, do this before requirements.txt)
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu124

# then surya BEFORE other requirements (has strict transformers version)
pip install surya-ocr==0.9.3

# then everything else
pip install -r requirements.txt

# also need this system package
sudo apt install poppler-utils
```

**start qdrant and redis:**

```bash
docker-compose up -d
```

**verify GPU works:**

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## training the model

first generate training data:

```bash
python scripts/generate_data.py
# takes about 20 min
# outputs: data/training_data/train.jsonl (120k examples)
```

then train:

```bash
python scripts/train.py
# takes ~8 hours on 4050
# model saved to model/checkpoint/
```

---

## running

**backend:**
```bash
uvicorn api.main:app --reload
```

**frontend:**
```bash
cd frontend
npm install
npm run dev
```

open `http://localhost:5173`

**quick test without frontend:**
```bash
python main.py some_legal_doc.pdf
```

---

## running tests

```bash
pytest tests/ -v
```

surya is mocked in tests so you don't need a GPU to run them.

---

## current status

- [x] OCR pipeline (pymupdf + surya)
- [ ] ingest IPC/BNS/Constitution into qdrant
- [ ] fine-tune InLegalBERT (need to generate data first)
- [ ] FastAPI + celery async jobs
- [ ] React frontend with PDF canvas
- [ ] Docker everything together

---

## stuff i learned building this

- pymupdf and fitz are the same thing (fitz is the old name)
- if you have a `frontend/` folder in your project root it conflicts with pymupdf's internal module... had to use `import pymupdf as fitz` to fix it
- surya 0.20.0 is completely different from 0.9.3, newer version needs a whole separate server to run. using 0.9.3
- surya default batch sizes (256 for recognition) are way too big for 6GB VRAM, had to drop to 32
- InLegalBERT is a BERT model pre-trained specifically on Indian legal text which is nice

---

## dependencies and why

| package | why |
|---|---|
| pymupdf | reads text from PDFs with exact word positions |
| surya-ocr | OCR for scanned PDFs, works well with Hindi |
| InLegalBERT | BERT model already trained on Indian legal text |
| qdrant | vector database for storing IPC/BNS sections |
| fastapi | backend API |
| celery + redis | PDF processing is slow (30-60s), need async jobs |
| react + pdf.js | render PDF in browser and draw highlight boxes on top |

---

## known issues

- surya is slow (~10s per scanned page on 4050)
- sometimes gets OOM on pages with lots of dense text, falls back to native extractor
- semantic error detection (wrong section numbers) isn't great yet, needs more training data
- frontend is basically empty right now

---

## references

- [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)
- [surya OCR](https://github.com/VikParuchuri/surya)
- [IL-TUR dataset](https://huggingface.co/datasets/rceborg/il-tur-lsi)
- [IndiaCode](https://indiacode.nic.in) - source for IPC, BNS, Constitution PDFs