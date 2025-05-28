import os

from pyvis.network import Network


def make_pyvis(tuples, node_color, edge_color, filename, title):
    """
    Build and save a PyVis interactive graph for given tuples.
    Exactly the same parameters as before.
    """
    # write into ./spice_cache
    cache_dir = os.path.join(os.getcwd(), "spice_cache")
    os.makedirs(cache_dir, exist_ok=True)

    net = Network(
        height="500px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
        cdn_resources="remote",
    )
    # tighter/shorter branches
    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.5,
        spring_length=80,
        spring_strength=0.1,
        damping=0.4,
        overlap=0.2,
    )
    net.toggle_physics(True)

    for tpl in tuples:
        if len(tpl) == 1:
            net.add_node(
                tpl[0],
                label=tpl[0],
                color={"border": "black", "background": node_color},
                size=20,
            )
        else:
            for src, dst in zip(tpl, tpl[1:]):
                net.add_node(
                    src,
                    label=src,
                    color={"border": "black", "background": node_color},
                    size=20,
                )
                net.add_node(
                    dst,
                    label=dst,
                    color={"border": "black", "background": node_color},
                    size=20,
                )
                net.add_edge(
                    src,
                    dst,
                    color=edge_color,
                    arrows="to",
                    smooth={"type": "curvedCCW", "roundness": 0.15},
                )

    # net.show_buttons(filter_=['physics'])
    # net.set_options("""
    # {
    #   "nodes": {
    #     "borderWidth": 4,
    #     "borderWidthSelected": 6,
    #     "font": {"size": 14}
    #   },
    #   "edges": {
    #     "smooth": {"enabled": true}
    #   },
    #   "interaction": {
    #     "hover": true,
    #     "navigationButtons": true,
    #     "zoomView": false
    #   }
    # }
    # """)

    out_path = os.path.join(cache_dir, filename)
    net.write_html(out_path, local=True)
    with open(out_path, "r", encoding="utf-8") as f:
        return f.read()
