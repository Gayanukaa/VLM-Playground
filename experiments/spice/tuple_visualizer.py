import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

def draw_scene_graph(tuples: list[tuple[str,str,str]]):
    """Given a list of (object, relation, object), render a simple graph."""
    G = nx.DiGraph()
    for subj, pred, obj in tuples:
        G.add_node(subj); G.add_node(obj)
        G.add_edge(subj, obj, label=pred)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1500, font_size=10)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    st.pyplot(plt.gcf())