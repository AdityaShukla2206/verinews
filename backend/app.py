"""VeriNews backend. Run: python app.py  ->  http://127.0.0.1:5000
Pipeline: language detect -> threat scanners -> ML classifier -> KB retrieval -> optional Google Fact Check -> verdict."""
import json, os, re, sqlite3, time, uuid
from pathlib import Path
from urllib.parse import urlparse
import joblib, requests
from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024
FC_KEY = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
KB = json.loads((BASE / "data" / "knowledge_base.json").read_text(encoding="utf8"))
MODEL_PATH = BASE / "models" / "model.joblib"
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
DB = BASE / "data" / "verinews.db"
KB_VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
KB_MAT = KB_VEC.fit_transform([k["headline"] + " " + k["claim"] for k in KB])
KB_THRESHOLD = 0.45


def db():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS submissions(id TEXT PRIMARY KEY, ts REAL, platform TEXT, text TEXT, label TEXT)")
    return c


# ---------- scanners ----------
BAD_TLD = {"xyz", "top", "click", "info", "live", "icu", "work", "site", "online", "buzz", "vip"}
SHORT = {"bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "is.gd", "goo.gl", "t.ly"}
BRANDS = ["sbi", "hdfc", "icici", "axis", "pnb", "paytm", "kotak", "irctc", "pm-", "gov", "aadhaar", "epfo", "itr"]
OFFICIAL = ("gov.in", "nic.in", "sbi.co.in", "hdfcbank.com", "icicibank.com", "axisbank.com", "paytm.com", "irctc.co.in")
HINGLISH = set("hai hain nahi karein kare jaldi sabko kal se band aapka aapke abhi warna bhejo bhejein khabar logon jeete paise "
               "batao bataiye jayega jayenge aaj baje saath ke ki ko par me mein ye is msg".split())
FLAGS = [  # (pattern, kind, weight, message)
    (r"\b(urgent|immediately|within \d+ ?(hours?|hrs)|last chance|only \d+ (left|remaining)|hurry|jaldi|turant|abhi|तुरंत)\b", "scam", 1,
     "Artificial urgency / deadline pressure"),
    (r"\b(won|winner|lottery|prize|jeete|jackpot|free (laptop|phone|smartphone|gift|tablet)|refund|cashback)\b", "scam", 1,
     "Prize / free-gift / refund lure"),
    (r"\b(processing fee|registration (fee|rs)|pay rs|delivery fee|pay .{0,20}fee)\b", "scam", 2, "Asks for an upfront fee"),
    (r"\b(account|card|sim|kyc|खाता).{0,40}\b(blocked|suspended|band|closed|frozen|बंद)\b|\b(blocked|suspended).{0,30}(account|card|sim)\b", "scam", 1,
     "Account-suspension threat"),
    (r"\b(forward|share)\b.{0,50}\b(all|everyone|groups?|friends|\d+ (logon|people|groups))\b|sabko forward|सभी को भेजें|forward (karein|karo)", "hoax", 2,
     "Chain-forward pressure (spreads by forwarding)"),
    (r"(media|hospitals?|government|doctors?) .{0,25}(hiding|secret|nahi bataega|नहीं बताएगा)|before (it is|they) (deleted|remove)|don'?t tell anyone", "hoax", 2,
     "'They are hiding this' conspiracy framing"),
    (r"\b(nasa|who|doctors|bank officials)\b.{0,20}\b(confirm|reveal|leaked|bataya)", "hoax", 1, "Unnamed/vague authority cited as proof"),
    (r"\b(100% sach|100% true|shocking|breaking!+)\b", "hoax", 1, "Sensational language"),
]
CRED = r"(share|send|enter|tell|provide|batao|bataiye|bhejein|बताएं).{0,40}\b(otp|cvv|pin|password|aadhaar|card number|pan)\b|\b(otp|cvv|card number|aadhaar)\b.{0,30}(at|on|to) "
SAFE_OTP = r"do not share|don't share|never share|kisi ko na batayein"


def scan(text):
    t, low = text, text.lower()
    scam = hoax = 0
    flags = []
    for pat, kind, w, msg in FLAGS:
        if re.search(pat, low, re.I):
            flags.append(msg)
            scam += w if kind == "scam" else 0
            hoax += w if kind == "hoax" else 0
    if re.search(CRED, low) and not re.search(SAFE_OTP, low):
        scam += 3; flags.append("Asks you to disclose OTP / CVV / card / Aadhaar details")
    domains = []
    for m in re.findall(r"(?:https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})(?:/\S*)?", low):
        if m.split(".")[-1] in {"in", "com", "org", "net", "co"} and m in ("e.g", "i.e"): continue
        domains.append(m)
    domains = list(dict.fromkeys(domains))
    for d in domains:
        tld, official = d.split(".")[-1], d.endswith(OFFICIAL)
        if d in SHORT: scam += 2; flags.append(f"URL shortener hides destination ({d})")
        elif tld in BAD_TLD: scam += 2; flags.append(f"Suspicious domain extension (.{tld}) in {d}")
        if not official and any(b in d.split(".")[0] for b in BRANDS):
            scam += 3; flags.append(f"Lookalike domain imitating a trusted brand: {d}")
        if d.count("-") >= 2: scam += 1; flags.append(f"Hyphen-stuffed domain: {d}")
    return scam, hoax, flags, domains


def language(text):
    if len(re.findall(r"[\u0900-\u097F]", text)) > 3: return "Hindi (Devanagari)"
    w = re.findall(r"[a-z]+", text.lower())
    return "Hinglish (code-mixed)" if sum(x in HINGLISH for x in w) >= 2 else "English"


def ml_probs(text):
    if not MODEL: return {}
    p = MODEL.predict_proba([text])[0]
    return {c: round(float(x), 3) for c, x in zip(MODEL.classes_, p)}


def kb_match(text):
    s = cosine_similarity(KB_VEC.transform([text]), KB_MAT)[0]
    i = int(s.argmax())
    return (KB[i], float(s[i])) if s[i] >= KB_THRESHOLD else (None, float(s[i]))


def fact_check(text):
    if not FC_KEY: return None, "Google Fact Check API key not configured on server."
    try:
        r = requests.get("https://factchecktools.googleapis.com/v1alpha1/claims:search",
                         params={"query": text[:450], "pageSize": 10, "key": FC_KEY}, timeout=8)
        r.raise_for_status()
    except Exception as e:
        return None, f"Fact Check API lookup failed: {type(e).__name__}"
    norm = lambda v: set(w for w in re.sub(r"[^\w]+", " ", v.lower()).split() if len(w) > 2)
    inp, best, bs = norm(text), None, 0
    for c in r.json().get("claims", []):
        for rv in c.get("claimReview", []):
            sc = len(inp & norm(c.get("text", ""))) / max(1, len(inp))
            if rv.get("textualRating") and sc > bs: best, bs = (c, rv), sc
    return (best if bs >= 0.35 else None), "No sufficiently close published fact-check found."


def rating_type(r):
    r = r.lower()
    if re.search(r"\b(false|fake|fabricated|incorrect|misleading|pants on fire)\b", r): return "FAKE"
    if re.search(r"\b(true|correct|accurate|verified)\b", r): return "REAL"
    return "UNVERIFIED"


def report(text, typ, **kw):
    d = dict(id="VN-" + uuid.uuid4().hex[:8].upper(), type=typ, headline=text, channel="User-submitted text",
             category="Claim requires independent fact-checking", language=language(text), verification="UNVERIFIED",
             summary="", claim=text, fact="", sources=[], redFlags=[], provenance="", domain="Not supplied", ip="Not applicable",
             pattern="", advice=[])
    d.update(kw)
    return d


@app.post("/api/analyze")
def analyze():
    t0 = time.perf_counter()
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()[:4000]
    if not text: return jsonify(error="text is required"), 400
    scam, hoax, flags, domains = scan(text)
    probs = ml_probs(text)
    top = max(probs, key=probs.get) if probs else "n/a"
    ml_line = f"ML classifier ({'TF-IDF+LogReg' if MODEL else 'not loaded'}): " + ", ".join(f"{k} {v:.0%}" for k, v in probs.items())
    entry, sim = kb_match(text)
    if entry:  # curated, source-backed record
        d = dict(entry); d["id"] = d["id"] + f"-M{int(sim*100)}"; d["headline"] = text
        d["redFlags"] = list(entry["redFlags"]) + [f"Registry similarity {sim:.0%}"] + flags
        d["engine"] = "curated-registry"
    else:
        fc, fc_msg = fact_check(text)
        if fc:
            c, rv = fc; typ = rating_type(rv["textualRating"])
            d = report(text, typ, category="Published ClaimReview match", verification="Published rating: " + rv["textualRating"],
                       summary="Closest fact-check record: " + c.get("text", ""), fact=f"{rv.get('publisher', {}).get('name', 'Publisher')} rated a matching claim: {rv['textualRating']}. Confirm that date and context match.",
                       sources=[f"{rv.get('publisher', {}).get('name', '')}: {rv.get('url', '')}", "Google Fact Check Tools API"],
                       redFlags=flags or ["No linguistic red flags raised."], provenance="Matched against a ClaimReview record.",
                       pattern="External fact-check match", advice=["Open the source link and read the full fact-check."])
            d["engine"] = "google-factcheck"
        elif scam >= 4 or (scam >= 2 and probs.get("SCAM", 0) >= 0.5) or probs.get("SCAM", 0) >= 0.8:
            d = report(text, "SCAM", category="Phishing / financial fraud pattern", verification="Pattern-based detection (scanners + ML)",
                       summary="Message shows multiple fraud indicators: " + "; ".join(flags[:3]) + ".",
                       fact="No legitimate bank, government body or courier asks for OTPs, fees or KYC via unsolicited links. Treat as fraud.",
                       sources=["National Cyber Crime Portal: https://cybercrime.gov.in (helpline 1930)", "CERT-In advisories: https://www.cert-in.org.in"],
                       redFlags=flags, provenance="Detected from linguistic + URL threat scanners and the trained classifier.",
                       domain=", ".join(domains) or "No URL found", pattern=f"Scam score {scam} / ML SCAM {probs.get('SCAM', 0):.0%}",
                       advice=["🚨 Do not click links or share OTP/CVV/PIN.", "🚨 Report on cybercrime.gov.in or call 1930."])
            d["engine"] = "scanners+ml"
        else:
            hoaxy = hoax >= 2 or probs.get("HOAX", 0) >= 0.6
            d = report(text, "UNVERIFIED",
                       summary=("Forward-chain misinformation style detected, but no published evidence was found. " if hoaxy else "No scam pattern and no matching fact-check record. ") + fc_msg,
                       fact="The system does not label a claim REAL or FAKE without an evidence record. Compare with primary sources (PIB Fact Check, official sites).",
                       sources=["PIB Fact Check: https://factcheck.pib.gov.in/", "Google Fact Check Explorer: https://toolbox.google.com/factcheck/explorer"],
                       redFlags=flags or ["No linguistic red flags raised."], provenance="Scanners + classifier ran; no evidence record matched.",
                       domain=", ".join(domains) or "No URL found", pattern="Hoax-style language pattern" if hoaxy else "No strong pattern",
                       advice=["Do not forward as confirmed until checked.", "Search the exact claim on PIB Fact Check or a reputed news outlet."])
            d["engine"] = "scanners+ml"
    d["ml"] = dict(probs=probs, top=top, line=ml_line)
    d["redFlags"] = d["redFlags"] + [ml_line]
    d["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return jsonify(d)


@app.get("/api/benchmarks")
def benchmarks():
    p = BASE / "models" / "metrics.json"
    return jsonify(json.loads(p.read_text())) if p.exists() else (jsonify(error="Run python train.py first"), 404)


@app.get("/api/kb")
def kb(): return jsonify(KB)


@app.post("/api/submit")
def submit():
    j = request.get_json(silent=True) or {}
    text = (j.get("text") or "").strip()[:2000]
    if len(text) < 10: return jsonify(error="Message too short"), 400
    sid = uuid.uuid4().hex[:10]
    with db() as c:
        c.execute("INSERT INTO submissions VALUES(?,?,?,?,NULL)", (sid, time.time(), str(j.get("platform", ""))[:40], text))
        n = c.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
    return jsonify(id=sid, total=n)


@app.get("/api/health")
def health():
    with db() as c: n = c.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
    return jsonify(ok=True, model_loaded=bool(MODEL), factcheck_configured=bool(FC_KEY), kb_entries=len(KB), submissions=n)


@app.get("/")
def index(): return send_from_directory(BASE.parent / "frontend", "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
