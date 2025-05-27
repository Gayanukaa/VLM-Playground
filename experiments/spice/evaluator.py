import os
import json
import subprocess
from nltk import download as nltk_download
from nltk.corpus import wordnet

# 1) Download needed NLTK data
nltk_download('wordnet', quiet=True)
nltk_download('omw-1.4', quiet=True)

# 2) Path to your SPICE jar
BASE_DIR   = os.path.dirname(__file__)
SPICE_JAR  = os.path.join(BASE_DIR, 'SPICE-1.0', 'spice-1.0.jar')

def _run_spice(input_json: str, output_json: str, detailed: bool = True) -> dict:
    """
    Invokes SPICE-1.0.jar on input_json, writes full JSON to output_json,
    and returns the single-result dict.
    """
    if not os.path.isfile(SPICE_JAR):
        raise FileNotFoundError(f"SPICE jar not found at {SPICE_JAR}")

    cmd = [
        'java', '-Xmx8G',
        '-jar', SPICE_JAR,
        input_json
    ]
    if detailed:
        cmd.append('-detailed')
    # write JSON to a file, suppress human-readable stdout
    cmd += ['-out', output_json, '-silent']

    proc = subprocess.run(cmd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"SPICE failed (exit {proc.returncode}):\n{proc.stderr}")

    # Load the JSON array and return its first element
    with open(output_json, 'r') as f:
        results = json.load(f)
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"Unexpected SPICE output format: {results}")
    return results[0]

def _wordnet_match(w1: str, w2: str) -> bool:
    """Return True if w1 == w2 or they share any WordNet synset."""
    if w1 == w2:
        return True
    syns1 = {lemma.name() for syn in wordnet.synsets(w1) for lemma in syn.lemmas()}
    syns2 = {lemma.name() for syn in wordnet.synsets(w2) for lemma in syn.lemmas()}
    return bool(syns1 & syns2)

def precision_recall_f1(hyp_tups: list, ref_tups: list) -> tuple:
    """Compute tuple-level P/R/F1 via binary WordNet matching."""
    matched = 0
    used = set()
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
    """Evaluates one candidate caption vs references with SPICE + WordNet-based tuple matching."""
    def __init__(self):
        # We'll write our temporary files here
        self.cache_dir = os.path.join(BASE_DIR, 'spice_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

    def evaluate(self, candidate: str, references: list) -> dict:
        # 1) Write SPICE input
        inp = os.path.join(self.cache_dir, 'spice_in.json')
        with open(inp, 'w') as f:
            json.dump([{
                "image_id": 0,
                "test":      candidate,
                "refs":      references
            }], f)

        # 2) Run SPICE
        outp = os.path.join(self.cache_dir, 'spice_out.json')
        result = _run_spice(input_json=inp, output_json=outp, detailed=True)

        # 3) Official SPICE F1
        spice_f1 = result['scores']['All']['f']

        # 4) Extract tuples
        test_block = result.get('test_tuples', [])
        ref_block  = result.get('ref_tuples', [])
        test_tups  = [entry['tuple'] for entry in test_block]
        ref_tups   = [entry['tuple'] for entry in ref_block]

        # 5) Compute WordNet-based tuple P/R/F1
        p, r, binary_f1 = precision_recall_f1(test_tups, ref_tups)

        return {
            'spice_f1':       spice_f1,
            'precision':      p,
            'recall':         r,
            'binary_f1':      binary_f1,
            'test_tuples':    test_tups,
            'ref_tuples':     ref_tups
        }
