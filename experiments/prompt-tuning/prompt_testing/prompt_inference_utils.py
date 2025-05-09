from unsloth import FastVisionModel  # FastLanguageModel for LLMs
from transformers import TextIteratorStreamer
import threading
from sentence_transformers import SentenceTransformer, util
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider   import Cider
from pycocoevalcap.spice.spice   import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import pandas as pd

print("🔄 Loading vision-language model...")
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)
model.eval()
print("✅ Model loaded successfully.")

print("🔄 Loading sentence transformer for scoring...")
scorer = SentenceTransformer("all-MiniLM-L6-v2").to("cuda")
print("✅ Sentence transformer loaded.")

# this code is specifically for dataset with multiple reference captions

def get_similarity_score(reference_captions, generated_caption):
    try:
        total_score = 0.0
        for caption in reference_captions:
            ref_embed = scorer.encode(caption, convert_to_tensor=True)
            gen_embed = scorer.encode(generated_caption, convert_to_tensor=True)
            score = util.cos_sim(gen_embed, ref_embed).item()
            total_score += score

        avg_score = total_score / len(reference_captions) if reference_captions else 0.0
        return avg_score

    except Exception as e:
        return 0.0


def score_per_image(refs, hypos):
    """
    refs: dict[int, List[str]]
    hypos: dict[int, List[str]]  (one hypo per id)
    returns: dict[id, {METEOR: float, CIDEr: float, SPICE: float}]
    """
    scorers = [
        (Meteor(), "METEOR"),
        (Cider(),  "CIDEr"),
        (Spice(),  "SPICE")
    ]

    ptb = PTBTokenizer()
    refs_wrapped  = {i:[{"caption":c} for c in caps] for i, caps in refs.items()}
    hypos_wrapped = {i:[{"caption":hypos[i][0]}]  for i in hypos}
    refs_tok  = ptb.tokenize(refs_wrapped)
    hypos_tok = ptb.tokenize(hypos_wrapped)

    all_scores = {}
    for scorer, name in scorers:
        avg_score, per_image_scores = scorer.compute_score(refs_tok, hypos_tok)

        for idx, img_id in enumerate(hypos_tok.keys()):
            all_scores.setdefault(img_id, {})

            if name == "SPICE":
                # per_image_scores[idx] is a dict of categories → {f,p,r}
                f_all = per_image_scores[idx].get("All", {}).get("f", 0.0)
                all_scores[img_id][name] = f_all
            else:
                # METEOR and CIDEr return floats per image
                all_scores[img_id][name] = per_image_scores[idx]

    return all_scores

def run_inference(image, model, tokenizer, instruction):
    print(f"🧠 Running inference with instruction: {instruction}")
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]

        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        print(f"📝 Tokenized prompt: {input_text[:100]}...")  # show a short preview

        inputs = tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        inputs.pop("token_type_ids", None)  # Pixtral models don’t need this

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generated_caption = ""

        print("🚀 Starting generation thread...")
        thread = threading.Thread(
            target=model.generate,
            kwargs={
                **inputs,
                "streamer": streamer,
                "max_new_tokens": 64,
                "use_cache": True,
                "temperature": 1.0,
                "min_p": 0.1
            }
        )
        thread.start()

        # Collect the tokens from the streamer
        for token in streamer:
            generated_caption += token

        print(f"✅ Generated caption: {generated_caption.strip()}")
        return generated_caption.strip()

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return ""

def evaluate_sample(prompts, sample, multiple_refs):
    """
    prompts: list of instructions to evaluate
    sample: the sample to evaluate (a row from the DataFrame)
    """
    print(f"\n🔍 Starting evaluation for sample with reference: {sample['caption']}")
    hypos = dict()
    cosine_scores = []
    if multiple_refs:
        ref_cap_list = sample['caption']
    else:
        ref_cap_list = [sample['caption']]
    refs = {i: ref_cap_list for i in range(len(prompts))}

    for i, prompt in enumerate(prompts):
        print(f"🧪 Evaluating instruction {i+1}/{len(prompts)}: '{prompt}'")
        pred = run_inference(sample['image'], model, tokenizer, prompt)
        print(f"🔹 Generated: {pred}")
        cos_score = get_similarity_score(ref_cap_list, pred)
        print(f"🔹 Semantic similarity: {cos_score:.4f}")
        cosine_scores.append(cos_score)
        hypos[i] = [pred]

    print("📊 Scoring predictions with COCO metrics...")
    coco_scores = score_per_image(refs, hypos)

    results = []
    for i, prompt in enumerate(prompts):
        res = {
            "reference_captions": " || ".join(ref_cap_list),
            "generated": hypos[i],
            "semantic_similarity": cosine_scores[i],
            "METEOR": coco_scores[i].get("METEOR", 0.0),
            "CIDEr": coco_scores[i].get("CIDEr", 0.0),
            "SPICE": coco_scores[i].get("SPICE", 0.0),
        }
        print(f"✅ Result for instruction {i+1}: {res}")
        results.append(res)

    return results


def evaluate_batch(prompts_list, val_data, indexes, multiple_refs=True):
    """
    prompts_list: list of instructions to evaluate
    val_data: DataFrame with ['image', 'caption'] columns,
    indexes: list of indexes to sample from val_data
    """
    print("🚀 Starting batch evaluation...")
    all_results = []

    for i, (index, prompts) in enumerate(zip(indexes, prompts_list)):
        print(f"\n📦 Evaluating sample {i+1}/{len(indexes)} at index {index}...")
        results = evaluate_sample(prompts, val_data[index],multiple_refs)
        for r in results:
            r["sample_index"] = index
        all_results.append(results)

    print("\n🔄 Transposing results by prompt...")
    transposed = list(map(list, zip(*all_results)))

    print(f"📁 Creating {len(transposed)} DataFrames (one per prompt)...")
    dfs = [pd.DataFrame(rows) for rows in transposed]

    print("✅ Batch evaluation complete!")
    return dfs
