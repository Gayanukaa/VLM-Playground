import os
import streamlit as st
import streamlit.components.v1 as components
from evaluator import SpiceEvaluator
from scene_graph_visualizer import make_pyvis

st.set_page_config(page_title="SPICE Eval Dashboard", layout="wide")
st.title("🎯 SPICE & Scene-Graph Evaluation")

# ─── Summary (always visible) ─────────────────────────────────────────────────
st.markdown("""
This dashboard provides an interactive way to evaluate image captions through:

- **SPICE Evaluation**: Invokes the SPICE-1.0 Java jar to compute official precision, recall, and F₁.
- **Interactive Scene Graphs**: Visualizes the extracted (object–relation–object and object–attribute) tuples as force-directed graphs.
""")

# ─── Sidebar Inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Caption Inputs")
st.sidebar.markdown("Developed by [Gayanukaa](https://gayanukaa.github.io)")

candidate = st.sidebar.text_area(
    "Candidate Caption",
    "a herd of giraffe standing next to each other"
)
refs_text = st.sidebar.text_area(
    "Reference Captions (one per line)",
    "\n".join([
        "a couple of giraffes that are walking around",
        "a herd of giraffe standing on top of a dirt field.",
        "Several smaller giraffes that are in an enclosure.",
        "The giraffes are walking in different directions outside.",
        "A giraffe standing next to three baby giraffes in a zoo exhibit."
    ])
)

if not st.sidebar.button("Evaluate"):
    st.sidebar.info("Enter captions and click **Evaluate**.")
    st.stop()

# Parse references
refs = [r.strip() for r in refs_text.splitlines() if r.strip()]
if not refs:
    st.sidebar.error("Please enter at least one reference caption.")
    st.stop()

# Run SPICE evaluation
evaluator = SpiceEvaluator()
out = evaluator.evaluate(candidate, refs)

# ─── Evaluation Results ─────────────────────────────────────────────────────────
st.subheader("📊 Evaluation Results")
c1, c2, c3 = st.columns(3)
c1.metric("Precision", f"{out['precision']:.3f}")
c2.metric("Recall",    f"{out['recall']:.3f}")
c3.metric("F₁-Score",  f"{out['spice_f1']:.3f}")

# ─── Extracted Tuples (Simplified) ──────────────────────────────────────────────
st.subheader("🔖 Extracted Tuples (Simplified)")
cand_tup_str = "[" + ", ".join(str(tuple(t)) for t in out['test_tuples']) + "]"
st.code(cand_tup_str, language="python")

# ─── Reference Tuples (Simplified) ──────────────────────────────────────────────
st.subheader("🔖 Reference Tuples (Simplified)")
for i, ref in enumerate(refs, start=1):
    # extract tuples for this single reference
    ref_out = evaluator.evaluate(ref, [ref])
    ref_tup_str = "[" + ", ".join(str(tuple(t)) for t in ref_out['test_tuples']) + "]"
    st.markdown(f"**Reference {i}:** “{ref}”")
    st.code(ref_tup_str, language="python")

# ─── Interactive Scene-Graphs ──────────────────────────────────────────────────
st.subheader("🌳 Interactive Scene-Graphs")
g1, g2 = st.columns(2)

with g1:
    st.markdown("**Candidate Caption Graph**")
    if out["test_tuples"]:
        html1 = make_pyvis(
            tuples     = out["test_tuples"],
            node_color = "lightblue",
            edge_color = "blue",
            filename   = "candidate_graph.html",
            title      = "Candidate"
        )
        components.html(html1, height=550, scrolling=True)
    else:
        st.warning("No candidate tuples to visualize.")

with g2:
    st.markdown("**Reference Caption Graph**")
    if out["ref_tuples"]:
        html2 = make_pyvis(
            tuples     = out["ref_tuples"],
            node_color = "lightgreen",
            edge_color = "green",
            filename   = "reference_graph.html",
            title      = "Reference"
        )
        components.html(html2, height=550, scrolling=True)
    else:
        st.warning("No reference tuples to visualize.")
