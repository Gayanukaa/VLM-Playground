import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

from evaluator import SpiceEvaluator

st.set_page_config(page_title="SPICE Tuple Evaluator", layout="wide")
st.title("🎯 SPICE + Tuple Visualizer")

# ─ Sidebar Inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Enter Captions")
candidate = st.sidebar.text_area("Candidate Caption", "A dog is running in the park.")
refs_text = st.sidebar.text_area(
    "Reference Captions (one per line)",
    "A dog runs outside.\nA canine is playing in the field."
)

if st.sidebar.button("Evaluate"):
    references = [r.strip() for r in refs_text.splitlines() if r.strip()]
    if not references:
        st.sidebar.error("Please provide at least one reference.")
        st.stop()

    # Run evaluation
    evaluator = SpiceEvaluator()
    result    = evaluator.evaluate(candidate, references)

    # ─ Metrics ────────────────────────────────────────────────────────────────
    st.subheader("📊 Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPICE F₁",       f"{result['spice_f1']:.3f}")
    c2.metric("Tuple Precision", f"{result['precision']:.3f}")
    c3.metric("Tuple Recall",    f"{result['recall']:.3f}")
    c4.metric("Tuple Binary F₁", f"{result['binary_f1']:.3f}")

    # ─ Raw Tuples ─────────────────────────────────────────────────────────────
    st.subheader("🔖 Extracted Tuples")
    col1, col2 = st.columns(2)
    col1.write("**Hypothesis Tuples**")
    col1.json(result['test_tuples'])
    col2.write("**Reference Tuples**")
    col2.json(result['ref_tuples'])

    # ─ Tuple Graph ────────────────────────────────────────────────────────────
    st.subheader("🌳 Tuple Graph")
    test_tups = result['test_tuples']
    ref_tups  = result['ref_tuples']

    if not (test_tups or ref_tups):
        st.warning("No tuples to visualize.")
    else:
        # Build a directed graph
        G = nx.DiGraph()
        # Hypothesis in green
        for tpl in test_tups:
            if len(tpl) == 1:
                G.add_node(tpl[0], color='green')
            else:
                for u, v in zip(tpl, tpl[1:]):
                    G.add_edge(u, v, color='green')
        # Reference in blue, overlap in purple
        for tpl in ref_tups:
            if len(tpl) == 1:
                if G.has_node(tpl[0]):
                    G.nodes[tpl[0]]['color'] = 'purple'
                else:
                    G.add_node(tpl[0], color='blue')
            else:
                for u, v in zip(tpl, tpl[1:]):
                    if G.has_edge(u, v):
                        G[u][v]['color'] = 'purple'
                    else:
                        G.add_edge(u, v, color='blue')

        pos         = nx.spring_layout(G, seed=42)
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        node_colors = [data.get('color', 'gray') for _, data in G.nodes(data=True)]

        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            arrowsize=20,
            ax=ax
        )
        st.pyplot(fig)

else:
    st.info("Enter your captions in the sidebar and click **Evaluate**.")
