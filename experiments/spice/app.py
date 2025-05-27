import streamlit as st
from evaluator import SpiceEvaluator

st.title("SPICE Scene‐Graph & Tuple Evaluator")

cand = st.text_area(
    "Candidate Caption",
    value="A dog is running in the park."
)
refs = st.text_area(
    "Reference Captions (one per line)",
    value="A dog runs outside.\nA canine is playing in the field."
)

if st.button("Evaluate"):
    refs_list = [r.strip() for r in refs.splitlines() if r.strip()]
    evaluator = SpiceEvaluator(cache_dir='spice_cache')
    out = evaluator.evaluate(cand, refs_list)

    st.subheader("Scores")
    st.write(f"🟢 SPICE F₁ (official): **{out['spice_f1']:.3f}**")
    st.write(f"🔵 Precision (tuple‐level): **{out['precision']:.3f}**")
    st.write(f"🔵 Recall (tuple‐level): **{out['recall']:.3f}**")
    st.write(f"🔵 Binary F₁ (tuple‐match): **{out['binary_f1']:.3f}**")

    st.subheader("Extracted Tuples")
    st.markdown("**Hypothesis Tuples:**")
    st.write(out['hypothesis_tuples'])
    st.markdown("**Reference Tuples:**")
    st.write(out['reference_tuples'])
