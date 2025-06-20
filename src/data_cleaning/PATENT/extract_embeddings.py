import pickle as pkl
import json

with open("prova/data2/edges.jsonl", "rb") as f:
    edges = [json.loads(l)["response"]["body"]["data"][0]["embedding"] for l in f.readlines()]

with open("prova/data2/hyperedge_embeddings.pkl", "wb") as f:
    pkl.dump(edges, f)

tot_nodes = []
for i in range(6):
    with open(f"prova/data2/batch_{i}_output.jsonl", "rb") as f:
        nodes = [json.loads(l)["response"]["body"]["data"][0]["embedding"] for l in f.readlines()]
        tot_nodes.extend(nodes)

with open("prova/data2/node_embeddings.pkl", "wb") as f:
    pkl.dump(tot_nodes, f)

