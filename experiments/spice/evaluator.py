import os
import json
import subprocess
from nltk.corpus import wordnet
from nltk import download as nltk_download

# Ensure WordNet data is available
nltk_download('wordnet', quiet=True)
nltk_download('omw-1.4', quiet=True)

# Paths to the SPICE distribution
BASE_DIR  = os.path.dirname(__file__)
SPICE_DIR = os.path.join(BASE_DIR, 'SPICE-1.0')
SPICE_JAR = os.path.join(SPICE_DIR, 'spice-1.0.jar')

def _run_spice(input_json: str,
               output_json: str,
               cache_dir: str = None,
               detailed: bool = True) -> dict:
    """
    Invoke the SPICE jar; write JSON output to `output_json`, then load & return
    the single-result dict.
    """
    cmd = [
        'java', '-Xmx8G',
        '-jar', SPICE_JAR,
        input_json
    ]
    if detailed:
        cmd.append('-detailed')
    if cache_dir:
        cmd += ['-cache', cache_dir]
    # direct SPICE to write its full JSON to a file, suppress stdout summary
    cmd += ['-out', output_json, '-silent']

    print(f"DEBUG: Running SPICE command:\n    {' '.join(cmd)}")
    proc = subprocess.run(cmd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True)
    print("DEBUG: SPICE stderr:\n", proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"SPICE failed (exit {proc.returncode}):\n{proc.stderr}")

    with open(output_json, 'r') as f:
        results = json.load(f)

    if not isinstance(results, list) or len(results) != 1:
        raise ValueError(f"Expected single-element list from SPICE, got: {results}")
    return results[0]

def _wordnet_match(w1: str, w2: str) -> bool:
    """
    Return True if w1 == w2 or they share any WordNet synset.
    """
    if w1 == w2:
        return True
    syns1 = {lemma.name() for syn in wordnet.synsets(w1) for lemma in syn.lemmas()}
    syns2 = {lemma.name() for syn in wordnet.synsets(w2) for lemma in syn.lemmas()}
    return bool(syns1 & syns2)

def precision_recall_f1(hyp_tuples: list, ref_tuples: list) -> tuple:
    """
    Compute tuple-level precision, recall, and F1 using binary WordNet matching.
    """
    matched = 0
    used_idx = set()

    for h in hyp_tuples:
        for idx, r in enumerate(ref_tuples):
            if idx in used_idx:
                continue
            if all(_wordnet_match(str(h[i]), str(r[i])) for i in range(len(h))):
                matched += 1
                used_idx.add(idx)
                break

    p = matched / len(hyp_tuples) if hyp_tuples else 0.0
    r = matched / len(ref_tuples) if ref_tuples else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1

class SpiceEvaluator:
    """
    Evaluates a single candidate caption against references using SPICE.
    Returns both the official SPICE F1 score and tuple-level precision/recall/F1.
    """
    def __init__(self, cache_dir: str = 'spice_cache'):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def evaluate(self, candidate: str, references: list) -> dict:
        # DEBUG: start
        print(f"DEBUG: Evaluating candidate='{candidate}' against {len(references)} references")

        # 1) Prepare input JSON
        inp_path = os.path.join(self.cache_dir, 'spice_inp.json')
        with open(inp_path, 'w') as f:
            json.dump([{
                "image_id": 0,
                "test": candidate,
                "refs": references
            }], f)
        print(f"DEBUG: Wrote SPICE input to {inp_path}")

        # 2) Run SPICE
        out_path = os.path.join(self.cache_dir, 'spice_out.json')
        result = _run_spice(
            input_json=inp_path,
            output_json=out_path,
            cache_dir=self.cache_dir,
            detailed=True
        )

        # 3) Debug top-level keys
        print(f"DEBUG: Loaded SPICE result: {result}")
        print("DEBUG: Top-level result keys:", list(result.keys()))

        # 4) Official SPICE F1 score
        if 'scores' not in result or 'All' not in result['scores']:
            raise KeyError(f"Missing 'scores' or 'scores[\"All\"]' in SPICE output: {result.get('scores')}")
        spice_f1 = result['scores']['All']['f']

        # 5) Extract test/reference tuples
        if 'test_tuples' not in result or 'ref_tuples' not in result:
            raise KeyError(f"Missing 'test_tuples' or 'ref_tuples' in SPICE output")
        hyp_block = result['test_tuples']
        ref_block = result['ref_tuples']
        print(f"DEBUG: Found {len(hyp_block)} test_tuples and {len(ref_block)} ref_tuples")

        hyp_tups = [entry['tuple'] for entry in hyp_block]
        ref_tups = [entry['tuple'] for entry in ref_block]

        # 6) Compute tuple-level P/R/F1
        p, r, binary_f1 = precision_recall_f1(hyp_tups, ref_tups)

        return {
            'spice_f1': spice_f1,
            'precision': p,
            'recall': r,
            'binary_f1': binary_f1,
            'hypothesis_tuples': hyp_tups,
            'reference_tuples': ref_tups
        }
