#!/usr/bin/env python3
"""Plot sampled graph walks on a 3D Earth globe."""
import argparse
import numpy as np
import plotly.graph_objects as go

from graph_transformer.data import GraphData, WalkDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("graph_dir")
    p.add_argument("--output", default="graph_walks_globe.html")
    p.add_argument("--walks", type=int, default=25)
    p.add_argument("--walk-length", type=int, default=8)
    p.add_argument("--k", type=int, default=None,
                   help="Only an annotation; graph edges are loaded from graph_dir")
    p.add_argument("--p", type=float, default=1.0)
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    graph = GraphData.load(args.graph_dir)
    dataset = WalkDataset(
        graph, length=args.walk_length, walks=args.walks,
        p=args.p, q=args.q, seed=args.seed,
    )

    sampled = []
    for i in range(args.walks):
        nodes = dataset._walk()
        coords = graph.coords[np.asarray(nodes)]
        sampled.append(coords)

    all_coords = np.concatenate(sampled)
    print("sampled latitude range:", float(all_coords[:, 0].min()), float(all_coords[:, 0].max()))
    print("sampled longitude range:", float(all_coords[:, 1].min()), float(all_coords[:, 1].max()))
    print("first sampled walk [lat, lon]:")
    print(sampled[0])

    # Plotly's built-in Natural Earth rendering supplies land and coastlines,
    # avoiding Cartopy/geopandas dependencies and local shapefile downloads.
    fig = go.Figure()
    colors = [
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
        "#17becf", "#e377c2", "#8c564b", "#bcbd22", "#7f7f7f",
    ]
    for i, coords in enumerate(sampled):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scattergeo(
            lat=coords[:, 0], lon=coords[:, 1],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(color=color, size=5),
            name=f"walk {i}",
            hovertemplate="walk %{text}<br>lat=%{lat:.4f}<br>lon=%{lon:.4f}<extra></extra>",
            text=[str(i)] * len(coords),
        ))
        fig.add_trace(go.Scattergeo(
            lat=[coords[0, 0]], lon=[coords[0, 1]],
            mode="markers",
            marker=dict(color="black", size=9, symbol="star"),
            name=f"start {i}",
            showlegend=False,
            hovertemplate="start of walk %{text}<br>lat=%{lat:.4f}<br>lon=%{lon:.4f}<extra></extra>",
            text=[str(i)],
        ))

    fig.update_layout(
        title=f"{args.walks} sampled graph walks",
        geo=dict(
            projection_type="orthographic",
            showland=True,
            landcolor="rgb(218, 225, 210)",
            showocean=True,
            oceancolor="rgb(205, 225, 245)",
            showcountries=True,
            countrycolor="rgba(80,80,80,0.45)",
            showcoastlines=True,
            coastlinecolor="rgb(50,50,50)",
            coastlinewidth=1.0,
            showlakes=True,
            lakecolor="rgb(190, 220, 245)",
         ),
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(itemsizing="constant"),
    )
    fig.write_html(args.output, include_plotlyjs=True)
    print("saved", args.output)


if __name__ == "__main__":
    main()
