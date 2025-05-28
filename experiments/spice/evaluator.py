import json
import os
import subprocess

from nltk import download as nltk_download
from nltk.corpus import wordnet

# Ensure WordNet is available
nltk_download("wordnet", quiet=True)
nltk_download("omw-1.4", quiet=True)

BASE_DIR = os.path.dirname(__file__)
SPICE_JAR = os.path.join(BASE_DIR, "SPICE-1.0", "spice-1.0.jar")


def _run_spice(input_json: str, output_json: str, detailed: bool, log_fn=None) -> dict:
    if not os.path.isfile(SPICE_JAR):
        raise FileNotFoundError(f"SPICE jar not found at {SPICE_JAR}")

    cmd = ["java", "-Xmx8G", "-jar", SPICE_JAR, input_json]
    if detailed:
        cmd.append("-detailed")
    cmd += ["-out", output_json, "-silent"]

    if log_fn:
        log_fn(f"> {' '.join(cmd)}")

    # Stream output line by line
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        if log_fn:
            log_fn(line.rstrip())
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"SPICE exited {ret}")

    # Load JSON
    with open(output_json, "r") as f:
        results = json.load(f)
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"Unexpected SPICE output: {results}")
    return results[0]


def _wordnet_match(w1: str, w2: str) -> bool:
    if w1 == w2:
        return True
    syns1 = {l.name() for s in wordnet.synsets(w1) for l in s.lemmas()}
    syns2 = {l.name() for s in wordnet.synsets(w2) for l in s.lemmas()}
    return bool(syns1 & syns2)


def precision_recall_f1(hyp_tups: list, ref_tups: list) -> tuple:
    matched, used = 0, set()
    for h in hyp_tups:
        for i, r in enumerate(ref_tups):
            if i in used:
                continue
            if all(_wordnet_match(str(h[j]), str(r[j])) for j in range(len(h))):
                matched += 1
                used.add(i)
                break
    p = matched / len(hyp_tups) if hyp_tups else 0.0
    r = matched / len(ref_tups) if ref_tups else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


class SpiceEvaluator:
    def __init__(self, cache_dir="spice_cache"):
        self.cache_dir = os.path.join(BASE_DIR, cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def evaluate(self, candidate: str, references: list, log_fn=None) -> dict:
        if log_fn:
            log_fn("🔧 Preparing input JSON...")
        inp = os.path.join(self.cache_dir, "inp.json")
        with open(inp, "w") as f:
            json.dump([{"image_id": 0, "test": candidate, "refs": references}], f)
        if log_fn:
            log_fn(f"✅ Wrote input to {inp}")

        outp = os.path.join(self.cache_dir, "out.json")
        if log_fn:
            log_fn("🚀 Running SPICE...")
        result = _run_spice(inp, outp, detailed=True, log_fn=log_fn)
        if log_fn:
            log_fn(f"✅ SPICE output in {outp}")

        # Extract SPICE metrics
        sc = result["scores"]["All"]
        sp_p = sc.get("pr", 0.0)
        sp_r = sc.get("re", 0.0)
        sp_f = sc.get("f", 0.0)
        if log_fn:
            log_fn(
                f"📊 SPICE -> Precision: {sp_p:.3f}, Recall: {sp_r:.3f}, F1: {sp_f:.3f}"
            )

        # Extract tuples
        test_block = result.get("test_tuples", [])
        ref_block = result.get("ref_tuples", [])
        test_tups = [e["tuple"] for e in test_block]
        ref_tups = [e["tuple"] for e in ref_block]
        if log_fn:
            log_fn(
                f"🌳 Parsed {len(test_tups)} test tuples and {len(ref_tups)} reference tuples"
            )

        # Compute tuple-level binary F1 (optional)
        p, r, bf1 = precision_recall_f1(test_tups, ref_tups)
        if log_fn:
            log_fn(f"✅ Tuple-level binary F1: {bf1:.3f}")

        return {
            "spice_precision": sp_p,
            "spice_recall": sp_r,
            "spice_f1": sp_f,
            "precision": p,
            "recall": r,
            "binary_f1": bf1,
            "test_tuples": test_tups,
            "ref_tuples": ref_tups,
        }
