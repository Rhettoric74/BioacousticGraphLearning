#!/usr/bin/env python3
"""Plot sampled graph walks on a 3D Earth globe."""
import argparse
import numpy as np
import matplotlib.pyplot as plt

from graph_transformer.data import GraphData, WalkDataset


def xyz(coords):
    lat = np.deg2rad(coords[:, 0])
    lon = np.deg2rad(coords[:, 1])
    return (np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("graph_dir")
    p.add_argument("--output", default="graph_walks_globe.png")
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

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(-np.pi / 2, np.pi / 2, 40)
    sphere_x = np.outer(np.cos(u), np.cos(v))
    sphere_y = np.outer(np.sin(u), np.cos(v))
    sphere_z = np.outer(np.ones_like(u), np.sin(v))
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color="lightsteelblue", alpha=0.16, linewidth=0)

    colors = plt.cm.tab20(np.linspace(0, 1, max(args.walks, 2)))
    for i, coords in enumerate(sampled):
        x, y, z = xyz(coords)
        ax.plot(x, y, z, color=colors[i % len(colors)], alpha=0.75, linewidth=1.5)
        ax.scatter(x, y, z, color=colors[i % len(colors)], s=14, alpha=0.9)
        ax.scatter(x[0], y[0], z[0], color="black", s=28, depthshade=False)

    ax.set_title(f"{args.walks} sampled graph walks")
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print("saved", args.output)


if __name__ == "__main__":
    main()
