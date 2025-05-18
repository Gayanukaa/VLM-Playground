# spice_evaluator.py

import os
import json
import tempfile
import shutil
import subprocess

import numpy as np
import pycocoevalcap.spice.spice as spice_mod
from pycocoevalcap.spice.spice import get_stanford_models

# Download CoreNLP models on first import
get_stanford_models()

class SpiceEvaluator:
    def __init__(self):
        # Locate the SPICE JAR and workspace dirs
        spice_dir      = os.path.dirname(spice_mod.__file__)
        self.jar_path  = os.path.join(spice_dir, spice_mod.SPICE_JAR)       # SPICE_JAR = 'spice-1.0.jar'  [oai_citation:0‡GitHub](https://raw.githubusercontent.com/salaniz/pycocoevalcap/master/spice/spice.py)
        self.temp_dir  = os.path.join(spice_dir, spice_mod.TEMP_DIR)         # TEMP_DIR = 'tmp'
        self.cache_dir = os.path.join(spice_dir, spice_mod.CACHE_DIR)        # CACHE_DIR = 'cache'

    def compute_spice(self, hypothesis: str, references: list[str]) -> dict:
        """
        Runs SPICE on a single hypothesis against multiple references.
        Returns a dict with overall score and per‐category breakdown.
        """
        avg_score, scores_list = self._run_spice(
            gts={0: references},
            res={0: [hypothesis]}
        )
        return {
            'spice_score': avg_score,
            'sub_scores': scores_list[0]
        }

    def _run_spice(self, gts: dict, res: dict):
        # Ensure temp & cache folders
        os.makedirs(self.temp_dir,  exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Build the JSON input array
        input_data = []
        for img_id in sorted(gts.keys()):
            hypo = res[img_id]
            ref  = gts[img_id]
            assert isinstance(hypo, list) and len(hypo) == 1
            assert isinstance(ref,  list) and len(ref)  >= 1
            input_data.append({
                'image_id': img_id,
                'test':      hypo[0],
                'refs':      ref
            })

        # Write input to a temp file (text mode)
        in_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, dir=self.temp_dir,
            suffix='.json', encoding='utf-8'
        )
        json.dump(input_data, in_file, indent=2)
        in_file.flush(); in_file.close()

        # Prepare output placeholder
        out_file = tempfile.NamedTemporaryFile(
            delete=False, dir=self.temp_dir, suffix='.json'
        )
        out_file.close()

        # Java command with module‐opens patch
        cmd = [
            'java',
            '--add-opens', 'java.base/java.lang=ALL-UNNAMED',
            '-Xmx8G',
            '-jar', self.jar_path,
            in_file.name,
            '-cache', self.cache_dir,
            '-out',   out_file.name,
            '-subset','-silent'
        ]

        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(self.jar_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if proc.returncode != 0:
            # Print out the Java error for debugging
            print("=== SPICE Java stdout ===")
            print(proc.stdout)
            print("=== SPICE Java stderr ===")
            print(proc.stderr)
            # Now raise a more informative exception
            raise RuntimeError(
                f"SPICE JAR failed with exit code {proc.returncode}. "
                "See above for Java error output."
            )
        # Read results
        with open(out_file.name, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # Clean up
        os.remove(in_file.name)
        os.remove(out_file.name)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Aggregate scores
        all_f   = []
        per_img = []
        for item in results:
            scores = item['scores']
            all_f.append(float(scores['All']['f']))
            # convert every sub‐category to float
            per_img.append({
                cat: {k: float(v) for k, v in scores[cat].items()}
                for cat in scores
            })

        avg_score = float(np.mean(all_f))
        return avg_score, per_img