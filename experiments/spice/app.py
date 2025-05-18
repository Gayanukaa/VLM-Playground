# app.py

import streamlit as st
from spice_evaluator import SpiceEvaluator
from tuple_visualizer import draw_scene_graph
from utils import are_synonyms

st.set_page_config(page_title="SPICE Metric + Tuple Visualiser")
st.title("SPICE Metric + Tuple Visualiser")

# Sidebar inputs
st.sidebar.header("Inputs")
hyp = st.sidebar.text_area(
    "Hypothesis Caption",
    "A man riding a horse on the beach."
)
refs_text = st.sidebar.text_area(
    "Reference Captions (one per line)",
    "A person is on a horse by the shore.\nA horse rider on the sand."
)
refs = [r.strip() for r in refs_text.splitlines() if r.strip()]

if st.sidebar.button("Evaluate"):
    evaluator = SpiceEvaluator()
    result = evaluator.compute_spice(hyp, refs)

    # Display the overall SPICE score
    st.subheader("SPICE Score")
    st.write(f"{result['spice_score']:.3f}")

    # Display breakdown by SPICE sub-categories
    st.subheader("Metric Breakdown")
    breakdown = result['sub_scores']
    breakdown_table = [
        {
            "Category": cat,
            "Precision": f"{m['precision']:.2f}",
            "Recall":    f"{m['recall']:.2f}",
            "F1":        f"{m['f']:.2f}"
        }
        for cat, m in breakdown.items()
    ]
    st.table(breakdown_table)

    # If your evaluator still returns raw tuples, visualize them
    if 'tuples' in result:
        st.subheader("Extracted Tuples")
        tuples_data = result['tuples']
        tuple_table = [
            {
                "Subject":   t[3][0],
                "Relation":  t[3][1],
                "Object":    t[3][2],
                "Precision": f"{t[0]:.2f}",
                "Recall":    f"{t[1]:.2f}",
                "F1":        f"{t[2]:.2f}"
            }
            for t in tuples_data
        ]
        st.table(tuple_table)

        st.subheader("Scene Graph")
        triple_list = [t[3] for t in tuples_data]
        draw_scene_graph(triple_list)

    # Binary match precision & recall using WordNet synonyms
    st.subheader("Binary Match (WordNet Synonym)")
    for idx, ref in enumerate(refs, start=1):
        h_tokens = hyp.lower().split()
        r_tokens = ref.lower().split()
        matches = sum(
            1
            for w in h_tokens
            for v in r_tokens
            if w == v or are_synonyms(w, v)
        )
        prec = matches / len(h_tokens) if h_tokens else 0.0
        rec  = matches / len(r_tokens) if r_tokens else 0.0
        st.write(f"Ref #{idx}: Precision = {prec:.2f}, Recall = {rec:.2f}")