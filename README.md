# VeriNews AI v3.1

Flask backend + single-page frontend. Pipeline: language detect → URL/linguistic scanners → TF-IDF ML classifier → curated fact registry (similarity retrieval) → optional Google Fact Check API → verdict.

## Run
```
pip install -r requirements.txt
python backend/train.py        # trains models, writes backend/models/metrics.json (Benchmarks tab)
python backend/app.py          # open http://127.0.0.1:5000
python backend/tests/test_api.py   # 13 tests
```
Optional live fact-check: `export GOOGLE_FACTCHECK_API_KEY=...` before starting (key stays on the server).

## Honest limits (say these in the viva)
- Verdicts REAL/FAKE come only from evidence (curated registry or a published fact-check). Otherwise UNVERIFIED. The ML model only flags scam/hoax-style language.
- Dataset is a small synthetic template set (SCAM/HOAX/LEGIT); metrics use template-grouped CV and are indicative only. No LIAR/FakeNewsNet, no transformer training yet.
- Registry has 5 entries (backend/data/knowledge_base.json); add more with the same fields. Re-check source URLs and dates before presenting.
- Submissions are stored in SQLite (backend/data/verinews.db) as an unlabeled review queue; they are not yet fed into training.
- If the backend is off, index.html falls back to a limited offline demo.
