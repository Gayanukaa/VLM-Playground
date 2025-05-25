import json
import subprocess # To simulate calling the SPICE jar
import os
import tempfile

# This is a placeholder for the actual SPICE scoring script/jar
# In a real scenario, you would download/install the official SPICE code
# (e.g., from https://github.com/peteanderson80/SPICE or coco-caption)
# which includes a spice-1.0.jar and Python wrappers.

# --- Configuration (adjust these paths if you have SPICE set up) ---
# Path to the directory containing spice-1.0.jar and supporting files
SPICE_DIR = "./spice_lib" # Example: where you might put the SPICE jar and related scripts
SPICE_JAR = os.path.join(SPICE_DIR, "spice-1.0.jar")
# Directory for temporary files SPICE might create
TEMP_DIR = os.path.join(SPICE_DIR, "tmp")
CACHE_DIR = os.path.join(SPICE_DIR, "cache")
# --------------------------------------------------------------------

def prepare_spice_input_data(gts, res):
    """
    Prepares the input data in the format expected by the SPICE evaluation script.
    :param gts: Dictionary with image_id -> list of reference captions
    :param res: Dictionary with image_id -> list of candidate captions (usually just one)
    :return: List of dictionaries for JSON input
    """
    assert sorted(gts.keys()) == sorted(res.keys())
    input_data = []
    for img_id in gts.keys():
        assert isinstance(gts[img_id], list)
        assert len(gts[img_id]) >= 1
        assert isinstance(res[img_id], list)
        assert len(res[img_id]) == 1

        input_data.append({
            "image_id": img_id,
            "test": res[img_id][0], # Candidate caption
            "refs": gts[img_id]    # List of reference captions
        })
    return input_data

def run_spice_evaluation(input_data):
    """
    Simulates running the SPICE evaluation.
    In a real setup, this function would call the spice-1.0.jar.
    For this example, it returns a mock JSON output.
    """
    if not os.path.exists(SPICE_JAR):
        print(f"WARNING: {SPICE_JAR} not found. Using mock SPICE output.")
        print("To run actual SPICE, download the SPICE package (e.g., from coco-caption) and update SPICE_JAR path.")
        # Mock output structure based on actual SPICE results
        mock_results = []
        for item in input_data:
            mock_results.append({
                "image_id": item["image_id"],
                "scores": {
                    "All": {"f": 0.65, "p": 0.70, "r": 0.60},
                    "Object": {"f": 0.75, "p": 0.80, "r": 0.70},
                    "Attribute": {"f": 0.55, "p": 0.60, "r": 0.50},
                    "Relation": {"f": 0.60, "p": 0.65, "r": 0.55}
                }
            })
        return mock_results

    # --- Actual SPICE call (if SPICE_JAR is available) ---
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=TEMP_DIR, suffix='.json')
    json.dump(input_data, in_file)
    in_file.close() # Close it so the Java process can read it

    out_file_path = tempfile.NamedTemporaryFile(mode='w', delete=False, dir=TEMP_DIR, suffix='.json').name

    # Typical SPICE command
    spice_cmd = [
        'java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
        '-cache', CACHE_DIR,
        '-out', out_file_path,
        '-subset', # Only score images in the input file
        '-silent'
    ]

    print(f"Running SPICE command: {' '.join(spice_cmd)}")
    try:
        subprocess.check_call(spice_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        with open(out_file_path) as f:
            results = json.load(f)
    except subprocess.CalledProcessError as e:
        print(f"Error running SPICE: {e}")
        return None # Or raise error
    except FileNotFoundError:
        print(f"Java or SPICE JAR not found. Ensure Java is in PATH and SPICE_JAR path is correct.")
        return None
    finally:
        os.remove(in_file.name)
        if os.path.exists(out_file_path):
             os.remove(out_file_path)
    return results


def display_spice_scores(spice_output):
    """
    Parses and displays the SPICE scores from the (mock or real) output.
    """
    if not spice_output:
        print("No SPICE output to display.")
        return

    total_spice_f = 0
    num_items = 0

    for item in spice_output:
        img_id = item["image_id"]
        scores = item["scores"]
        print(f"\n--- Scores for Image ID: {img_id} ---")
        print(f"  Candidate Caption: {next(d['test'] for d in input_data if d['image_id'] == img_id)}") # Retrieve for display
        print(f"  Reference Captions: {next(d['refs'] for d in input_data if d['image_id'] == img_id)}")

        all_scores = scores.get("All", {})
        spice_f = all_scores.get("f", 0.0)
        precision = all_scores.get("p", 0.0)
        recall = all_scores.get("r", 0.0)

        print(f"  Overall SPICE (F1-score): {spice_f:.4f}")
        print(f"  Overall Precision:        {precision:.4f}")
        print(f"  Overall Recall:           {recall:.4f}")

        print("  Scores by category:")
        for category, cat_scores in scores.items():
            if category != "All":
                print(f"    {category}: P={cat_scores.get('p', 0):.2f}, R={cat_scores.get('r', 0):.2f}, F1={cat_scores.get('f', 0):.2f}")

        total_spice_f += spice_f
        num_items +=1

    if num_items > 0:
        average_spice_f = total_spice_f / num_items
        print(f"\n--- Average SPICE (F1-score) across all items: {average_spice_f:.4f} ---")


if __name__ == "__main__":
    # Example data: gts (ground truth reference) and res (candidate result)
    gts_data = {
        "img1": [
            "A man is riding a brown horse on a dirt road.",
            "A person on a horse is moving along a path."
        ],
        "img2": [
            "A black cat is sleeping on a red couch.",
            "A feline rests on a crimson sofa."
        ]
    }

    res_data = {
        "img1": ["A man rides a horse on a road."],
        "img2": ["The cat sleeps on the red sofa."]
    }

    input_data = prepare_spice_input_data(gts_data, res_data)
    print("--- Prepared SPICE Input (JSON format) ---")
    print(json.dumps(input_data, indent=2))
    print("\n")

    # This would call the actual SPICE evaluation
    spice_results = run_spice_evaluation(input_data)

    if spice_results:
        print("\n--- Parsed SPICE Output ---")
        display_spice_scores(spice_results)
    else:
        print("SPICE evaluation failed or was skipped.")