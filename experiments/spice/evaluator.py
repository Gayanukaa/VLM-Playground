import os
import json
import subprocess
from nltk.corpus import wordnet
from nltk import download as nltk_download

# Ensure WordNet data is available
nltk_download('wordnet', quiet=True)
nltk_download('omw-1.4', quiet=True)

SPICE_JAR = os.path.join(os.path.dirname(__file__), 'spice', 'spice-1.0.jar')
CORENLP_DIR = os.path.join(os.path.dirname(__file__), 'spice', 'corenlp')

def _run_spice(input_json, cache_dir=None, detailed=True):
    cmd = [
        'java', '-Xmx8G', '-jar', SPICE_JAR, input_json,
        '-threads', str(os.cpu_count())
    ]
    if detailed:
        cmd.append('-detailed')
    if cache_dir:
        cmd += ['-cache', cache_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)

def extract_tuples(scores_json):
    """
    Returns list of (object, relation, attribute) tuples for candidate and refs.
    """
    return scores_json['all_propositions']  # per SPICE detailed output

def _wordnet_match(w1, w2):
    # Quick check for identical strings
    if w1 == w2:
        return True
    # Check for any shared synset
    syns1 = set(s.name() for syn in wordnet.synsets(w1) for s in syn.lemmas())
    syns2 = set(s.name() for syn in wordnet.synsets(w2) for s in syn.lemmas())
    return len(syns1 & syns2) > 0

def precision_recall_f1(hyp_tuples, ref_tuples):
    matched = 0
    used = set()
    for h in hyp_tuples:
        for i, r in enumerate(ref_tuples):
            if i in used: continue
            if all(_wordnet_match(str(h[k]), str(r[k])) for k in range(len(h))):
                matched += 1
                used.add(i)
                break
    p = matched / len(hyp_tuples) if hyp_tuples else 0.0
    r = matched / len(ref_tuples) if ref_tuples else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1

class SpiceEvaluator:
    def __init__(self, cache_dir='spice_cache'):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def evaluate(self, candidate, references):
        # Prepare SPICE input format
        data = {
            "sentence_id": 0,
            "image_id": 0,
            "caption": candidate,
            "references": references
        }
        tmp = os.path.join(self.cache_dir, 'inp.json')
        with open(tmp, 'w') as f:
            json.dump([data], f)
        out = _run_spice(tmp, cache_dir=self.cache_dir, detailed=True)
        # SPICE F1 score:
        spice_f1 = out['scores'][0]['All']['f']
        hyp_tups = extract_tuples(out['proposition_info'][0]['All']['candidate'])
        ref_tups = []
        for ri in out['proposition_info'][0]['All']['references']:
            ref_tups.extend(extract_tuples(ri))
        p, r, f1 = precision_recall_f1(hyp_tups, ref_tups)
        return {
            'spice_f1': spice_f1,
            'precision': p,
            'recall': r,
            'binary_f1': f1,
            'hypothesis_tuples': hyp_tups,
            'reference_tuples': ref_tups
        }
