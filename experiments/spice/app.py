# app.py

import os
import json
import streamlit as st
from evaluator import SpiceEvaluator
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="Interactive SPICE Graphs", layout="wide")
st.title("🎯 SPICE Scene-Graph Interactive Visualizer")

# ─── Sidebar Inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Caption Inputs")
candidate = st.sidebar.text_area(
    "Candidate Caption",
    "A dog is running in the park."
)
refs_text = st.sidebar.text_area(
    "Reference Captions (one per line)",
    "A dog runs outside.\nA canine is playing in the field."
)

if st.sidebar.button("Evaluate"):
    # parse references
    refs = [r.strip() for r in refs_text.splitlines() if r.strip()]
    if not refs:
        st.sidebar.error("Enter at least one reference caption.")
        st.stop()

    # run SPICE + tuple matching
    evaluator = SpiceEvaluator()
    out = evaluator.evaluate(candidate, refs)

    # ─── Metrics Row ────────────────────────────────────────────────────────────
    st.subheader("📊 Evaluation Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPICE F₁",        f"{out['spice_f1']:.3f}")
    c2.metric("Tuple Precision", f"{out['precision']:.3f}")
    c3.metric("Tuple Recall",    f"{out['recall']:.3f}")
    c4.metric("Tuple Binary F₁", f"{out['binary_f1']:.3f}")

    # ─── Raw Tuples ─────────────────────────────────────────────────────────────
    st.subheader("🔖 Extracted Tuples")
    col_h, col_r = st.columns(2)
    col_h.write("**Hypothesis (Candidate) Tuples**")
    col_h.json(out["test_tuples"])
    col_r.write("**Reference Tuples**")
    col_r.json(out["ref_tuples"])

    # ─── Interactive Graphs ───────────────────────────────────────────────────────
    st.subheader("🌳 Interactive Scene-Graphs")
    g1, g2 = st.columns(2)

    def make_pyvis(tuples, node_color, edge_color, filename, title):
        """
        Build and save a PyVis interactive graph for given tuples.
        """
        net = Network(
            height="500px", width="100%",
            bgcolor="#ffffff", font_color="black",
            directed=True
        )
        net.barnes_hut()  # nice force layout

        # add nodes & edges
        for tpl in tuples:
            if len(tpl) == 1:
                net.add_node(tpl[0], label=tpl[0], color=node_color)
            else:
                # for (subj, rel, obj) or (obj, attr)
                for src, dst in zip(tpl, tpl[1:]):
                    # ensure nodes exist
                    net.add_node(src, label=src, color=node_color)
                    net.add_node(dst, label=dst, color=node_color)
                    net.add_edge(src, dst, label=str(tpl[1]), color=edge_color, arrows="to")

        # write out & return HTML
        out_path = os.path.join(evaluator.cache_dir, filename)
        net.save_graph(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            return f.read()

    # Candidate graph (blue)
    with g1:
        st.markdown("**Candidate Graph**")
        if out["test_tuples"]:
            html1 = make_pyvis(
                out["test_tuples"],
                node_color="lightblue",
                edge_color="blue",
                filename="candidate_graph.html",
                title="Candidate"
            )
            components.html(html1, height=550, scrolling=True)
        else:
            st.warning("No candidate tuples to visualize.")

    # Reference graph (green)
    with g2:
        st.markdown("**Reference Graph**")
        if out["ref_tuples"]:
            html2 = make_pyvis(
                out["ref_tuples"],
                node_color="lightgreen",
                edge_color="green",
                filename="reference_graph.html",
                title="Reference"
            )
            components.html(html2, height=550, scrolling=True)
        else:
            st.warning("No reference tuples to visualize.")

else:
    st.info("Enter captions in the sidebar and click **Evaluate**.")
