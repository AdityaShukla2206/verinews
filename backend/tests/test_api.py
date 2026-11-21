import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as A

c = A.app.test_client()
def run(t): return c.post("/api/analyze", json={"text": t}).get_json()

class T(unittest.TestCase):
    def test_kb_laptop(self): self.assertEqual(run("PM free laptop yojana 2026 register now, Rs 5000 allowance")["type"], "SCAM")
    def test_kb_spadex(self): self.assertEqual(run("ISRO launched SpaDeX satellites on PSLV-C60 on 30 December 2024")["type"], "REAL")
    def test_kb_currency(self): self.assertEqual(run("Kal se saare 500 ke note band ho jayenge, sabko forward karo")["type"], "FAKE")
    def test_new_scam(self): self.assertEqual(run("Dear user your HDFC card is blocked. Share OTP now or update at hdfc-secure-kyc.top")["type"], "SCAM")
    def test_legit_otp_not_scam(self): self.assertNotEqual(run("Your OTP for transaction of Rs 250 is 482913. Do not share it with anyone. - SBI")["type"], "SCAM")
    def test_meeting_not_scam(self): self.assertNotEqual(run("Meeting tomorrow at 11 in the conference room, please bring the report.")["type"], "SCAM")
    def test_unknown_claim_unverified(self): self.assertEqual(run("The prime minister will visit Japan next month for a trade summit.")["type"], "UNVERIFIED")
    def test_hoax_pattern_not_fake(self):
        r = run("BREAKING: Government will ban WhatsApp from tomorrow, share with all your groups before it is deleted!")
        self.assertEqual(r["type"], "UNVERIFIED"); self.assertIn("misinformation", r["summary"].lower())
    def test_empty(self): self.assertEqual(c.post("/api/analyze", json={"text": " "}).status_code, 400)
    def test_benchmarks(self): self.assertEqual(len(c.get("/api/benchmarks").get_json()["models"]), 3)
    def test_submit(self):
        self.assertEqual(c.post("/api/submit", json={"text": "short"}).status_code, 400)
        self.assertIn("id", c.post("/api/submit", json={"platform": "SMS", "text": "Test suspicious message for unit test"}).get_json())
    def test_factcheck_mock(self):
        class R:
            def raise_for_status(s): pass
            def json(s): return {"claims": [{"text": "Moon landing was staged by studio", "claimReview": [{"textualRating": "False", "url": "http://x.org/a", "publisher": {"name": "X"}}]}]}
        A.FC_KEY, orig = "k", A.requests.get
        A.requests.get = lambda *a, **k: R()
        try: self.assertEqual(run("Moon landing was staged by a studio")["type"], "FAKE")
        finally: A.FC_KEY, A.requests.get = "", orig
    def test_index(self): self.assertEqual(c.get("/").status_code, 200)

unittest.main(verbosity=1)
