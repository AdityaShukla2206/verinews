"""Train + evaluate VeriNews classifiers. Run: python train.py
Writes models/model.joblib (deployed LogReg) and models/metrics.json (shown in Benchmarks tab).
Dataset = template-generated seed set (SCAM / HOAX / LEGIT). Grouped CV (by template) keeps
test templates unseen. Numbers are NOT comparable to LIAR/FakeNewsNet benchmarks."""
import json, random, time, itertools
from pathlib import Path
import joblib, numpy as np
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

BASE = Path(__file__).parent
F = dict(
    bank=["SBI", "HDFC", "ICICI", "Axis Bank", "PNB", "Paytm", "Kotak"],
    tld=["xyz", "top", "info", "click", "com", "live"],
    amt=["25,00,000", "50000", "1 crore", "10,000", "75000"], small=["99", "199", "499", "250"],
    num=["9876501234", "8800123456", "7011223344"], gadget=["laptop", "smartphone", "tablet"],
    thing=["500 ke note", "UPI payments", "ATM cards", "2000 ke note", "petrol pumps"],
    act=["block all SIM cards", "ban WhatsApp", "freeze bank accounts", "stop UPI"],
    food=["garlic", "lemon water", "turmeric milk", "ginger tea"], disease=["cancer", "diabetes", "corona", "dengue"],
    n=["5", "7", "10", "15"], celeb=["a famous minister", "a cricketer", "an actor"],
    sat=["SpaDeX", "Aditya", "NISAR", "Chandrayaan"], date=["Monday", "30 December", "15 March", "Friday", "5 June"],
    rate=["6.5", "5.5", "7.2", "6.0"], bill=["Finance", "Education", "Telecom", "Agriculture"],
    city=["Delhi", "Mumbai", "Pune", "Jaipur", "Chennai"], time=["10", "11:30", "3", "4:15"],
    otp=["482913", "775201", "630918"], otp4=["4421", "8830", "1297"],
)
SCAM = [
    "URGENT: Your {bank} account is suspended. Update KYC now at {bank}-kyc-update.{tld} or lose access in 24 hours.",
    "Congratulations! You won Rs {amt} lottery. Click bit.ly/claim{n} and enter your bank details to claim the prize.",
    "Dear customer your {bank} card is blocked. Share OTP to unblock. Call {num} immediately.",
    "{bank} aapka account band ho jayega. Abhi {bank}-pan.{tld} par PAN update karein warna paise kat jayenge.",
    "Free {gadget} scheme for students, register at pm-free-{gadget}.{tld} and pay Rs {small} fee. Only 100 left hurry!",
    "Work from home earn Rs {amt} daily. Just pay registration Rs {small} and send screenshot on WhatsApp {num}.",
    "Your parcel is held at customs. Pay Rs {small} delivery fee at bit.ly/parcel{n} within 12 hours.",
    "आपका {bank} खाता बंद हो जाएगा। तुरंत {bank}-kyc.{tld} पर KYC अपडेट करें और OTP बताएं।",
    "Electricity bill unpaid, power will be cut tonight. Call {num} and pay at bill-pay-now.{tld} immediately.",
    "Income tax refund of Rs {amt} approved. Verify card number and CVV at itr-refund.{tld} to receive.",
    "KBC lottery winner! Aapne Rs {amt} jeete hain. Processing fee Rs {small} bhejein {num} par.",
    "Your SIM will be blocked today. Send Aadhaar and OTP to {num} to continue service.",
]
HOAX = [
    "Khabar aa rahi hai kal se {thing} band ho jayenge. Sabko forward karein jaldi.",
    "BREAKING: Government will {act} from tomorrow, share with all your groups before it is deleted!",
    "WhatsApp will start charging Rs {small} per month from tomorrow, forward this to {n} friends to stay free.",
    "Doctors reveal {food} cures {disease} in 3 days, hospitals are hiding this truth. Share before they remove it.",
    "NASA confirms internet will shut down for {n} days, store cash and food, forward to everyone.",
    "सरकार कल से {thing} बंद कर रही है, यह संदेश सभी को भेजें, मीडिया नहीं बताएगा।",
    "Shocking video: {celeb} said something the media is hiding. Forward to every group, share till it reaches everyone!",
    "Drinking hot water with {food} kills the virus in one day, WHO secretly confirms. Forward to save lives.",
    "Mobile tower radiation causes {disease}, ban coming tomorrow, forward this urgent news.",
    "Big news!! New law: everyone must {act} by tomorrow or pay a fine of Rs {amt}. Forward to all.",
    "Is msg ko {n} logon ko bhejo warna aapke saath bura hoga, ye 100% sach hai, doctors ne bataya.",
    "Don't tell anyone: {thing} will be banned from midnight, bank officials leaked the news, forward now.",
]
LEGIT = [
    "ISRO successfully launched the {sat} satellite on {date}, the agency said in a statement.",
    "The Reserve Bank of India kept the repo rate unchanged at {rate} percent in its policy review on {date}.",
    "Your OTP for transaction of Rs {small} is {otp}. Do not share it with anyone. - {bank}",
    "Rs {amt} credited to your {bank} account ending {otp4} on {date}. Available balance updated.",
    "Parliament passed the {bill} bill on {date} after a debate that lasted several hours, officials said.",
    "The weather department forecast heavy rain in {city} on {date} and advised commuters to plan travel.",
    "Meeting tomorrow at {time} in the conference room, please bring the quarterly report.",
    "Aaj ki meeting {time} baje hai, please report ke saath aana. Thanks.",
    "{city} metro services will run with extended hours on {date} for the cricket match, the operator said.",
    "Your order has been shipped and will arrive by {date}. Track it in the app. Thank you for shopping.",
    "The finance ministry released GST collection data showing growth of {rate} percent year on year.",
    "आज शाम {time} बजे घर आ जाना, खाना साथ में खाएंगे।",
]


class _D(dict):
    def __missing__(self, k): return "{" + k + "}"


def build(per=10, seed=7):
    rnd, X, y, g = random.Random(seed), [], [], []
    for label, temps in (("SCAM", SCAM), ("HOAX", HOAX), ("LEGIT", LEGIT)):
        for ti, t in enumerate(temps):
            seen = set()
            for _ in range(per * 3):
                s = t.format_map(_D({k: rnd.choice(v) for k, v in F.items()}))
                if s not in seen and len(seen) < per:
                    seen.add(s); X.append(s); y.append(label); g.append(f"{label}{ti}")
    return X, np.array(y), np.array(g)


def vec():
    return make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True),
                      TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True))


MODELS = {
    "TF-IDF + Linear SVM": lambda: LinearSVC(C=1.0),
    "TF-IDF + Logistic Regression": lambda: LogisticRegression(max_iter=2000, C=5.0),
    "TF-IDF + Naive Bayes": lambda: MultinomialNB(alpha=0.1),
}


def params(clf):
    for a in ("coef_", "feature_log_prob_"):
        if hasattr(clf, a): return int(getattr(clf, a).size)
    return 0


if __name__ == "__main__":
    X, y, g = build()
    X = np.array(X, dtype=object)
    rows = []
    for name, mk in MODELS.items():
        P = []
        for tr, te in GroupKFold(n_splits=4).split(X, y, g):
            m = make_pipeline(vec(), mk()).fit(X[tr], y[tr])
            pr = m.predict(X[te])
            p, r, f, _ = precision_recall_fscore_support(y[te], pr, average="macro", zero_division=0)
            P.append((p, r, f, accuracy_score(y[te], pr)))
        full = make_pipeline(vec(), mk()).fit(X, y)
        t0 = time.perf_counter()
        for s in X[:200]: full.predict([s])
        lat = (time.perf_counter() - t0) / 200 * 1000
        p, r, f, a = np.mean(P, axis=0) * 100
        rows.append(dict(name=name, precision=round(p, 1), recall=round(r, 1), f1=round(f, 1), accuracy=round(a, 1),
                         latency_ms=round(lat, 2), parameters=params(full[-1])))
        print(rows[-1])
        if name == "TF-IDF + Logistic Regression":
            joblib.dump(full, BASE / "models" / "model.joblib")
    out = dict(dataset=dict(samples=len(X), classes={c: int((y == c).sum()) for c in ("SCAM", "HOAX", "LEGIT")},
                            templates=len(set(g)), evaluation="4-fold GroupKFold by template (unseen templates in each test fold)",
                            note="Synthetic template-based seed set; indicative only, not a LIAR/FakeNewsNet benchmark."),
               deployed="TF-IDF + Logistic Regression", models=rows)
    (BASE / "models" / "metrics.json").write_text(json.dumps(out, indent=1))
    print("saved model + metrics")
