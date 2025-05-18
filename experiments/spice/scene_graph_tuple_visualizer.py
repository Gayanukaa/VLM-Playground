import networkx as nx
import matplotlib.pyplot as plt

def visualize_scene_graph_tuples(tuples, caption_name="Scene Graph"):
    """
    Visualizes scene graph tuples.
    Tuples can be of form:
    - (subject, attribute)
    - (subject, relation, object)
    """
    graph = nx.MultiDiGraph() # Use MultiDiGraph for potentially multiple relations between same nodes

    nodes = set()
    for tpl in tuples:
        if len(tpl) == 2: # Attribute
            obj, attr = tpl
            nodes.add(str(obj))
            nodes.add(f"{str(attr)} (attr)") # Distinguish attribute nodes visually
            graph.add_edge(str(obj), f"{str(attr)} (attr)", label="has_attr")
        elif len(tpl) == 3: # Relation
            subj, rel, obj = tpl
            nodes.add(str(subj))
            nodes.add(str(obj))
            graph.add_edge(str(subj), str(obj), label=str(rel))
        else:
            print(f"Warning: Skipping invalid tuple {tpl}")

    if not graph.nodes:
        print("No tuples to visualize.")
        return

    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(graph, k=0.9, iterations=50) # k controls distance between nodes

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=2000, node_color="skyblue", alpha=0.9)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, arrowstyle="->", arrowsize=15,
                           edge_color="gray", width=1.5, alpha=0.7)

    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold")

    # Draw edge labels (relations/attributes)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, font_color='red')

    plt.title(caption_name)
    plt.axis('off') # Turn off the axis
    plt.show()

if __name__ == "__main__":
    # Example scene graph tuples (these would ideally be extracted from a scene graph parser)
    # based on "A large brown dog is running in the green field."
    example_tuples1 = [
        ("dog", "large"),
        ("dog", "brown"),
        ("dog", "running_in", "field"),
        ("field", "green")
    ]
    visualize_scene_graph_tuples(example_tuples1, caption_name="Graph for: 'A large brown dog is running in the green field.'")

    # based on "A black cat sleeps on the red sofa."
    example_tuples2 = [
        ("cat", "black"),
        ("cat", "sleeps_on", "sofa"),
        ("sofa", "red")
    ]
    visualize_scene_graph_tuples(example_tuples2, caption_name="Graph for: 'A black cat sleeps on the red sofa.'")

    # based on a more complex caption: "A young girl in a pink dress throws a red ball to a happy boy."
    example_tuples3 = [
        ("girl", "young"),
        ("girl", "wears", "dress"), # Simplified, could be (girl, has_clothing, dress)
        ("dress", "pink"),
        ("girl", "throws", "ball"),
        ("ball", "red"),
        ("girl", "throws_to", "boy"), # Or (ball, destination, boy) depending on parser
        ("boy", "happy")
    ]
    visualize_scene_graph_tuples(example_tuples3, caption_name="Graph for: 'A young girl in a pink dress throws a red ball to a happy boy.'")