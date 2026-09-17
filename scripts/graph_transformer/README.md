# Geographic graph transformer

From the repository root, run `python scripts/train_graph_transformer.py GRAPH_DIR`.
The default objective combines species classification with location-token and Perch-embedding reconstruction. Use `--location-weight 0 --audio-weight 0` for the species-only baseline. `--p` and `--q` control node2vec return/in-out bias.
