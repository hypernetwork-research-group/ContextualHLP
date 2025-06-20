import pandas as pd
import numpy as np
from collections import defaultdict
import pickle as pkl
import hypernetx as hnx
import openai
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import json

with open("prova/data2/edges.pkl", "rb") as f:
    edges = pkl.load(f)

with open("prova/data2/batch_edges.jsonl", "w") as f:
    for i, (k, edge) in enumerate(edges.items()):
        data = {
            "custom_id": f"edge-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Provide a brief description of the patent identified by {k} focusing on its content, technical domain, and area of application. Do not mention or reference the inventor or assignee."
                    }
                ],
                "max_tokens": 500
            }
        }
        f.write(json.dumps(data) + "\n")

with open("prova/data2/nodes.pkl", "rb") as f:
    nodes = pkl.load(f)

with open("prova/data2/batch_nodes.jsonl", "w") as f:
    for i, (k, node) in enumerate(nodes.items()):
        data = {
            "custom_id": f"node-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Provide a concise overview of {k}, focusing on their areas of expertise, research interests, and fields of professional activity. Avoid mentioning specific patents, projects."
                    }
                ],
                "max_tokens": 500
            }
        }
        f.write(json.dumps(data) + "\n")

# Per edges
with open("prova/data2/batch_edges.jsonl", "rb") as f:
    batch_edges_file = client.files.create(file=f, purpose="batch")

batch_edges = client.batches.create(
    input_file_id=batch_edges_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "patent edges descriptions"}
)

# Per nodes
with open("prova/data2/batch_nodes.jsonl", "rb") as f:
    batch_nodes_file = client.files.create(file=f, purpose="batch")

batch_nodes = client.batches.create(
    input_file_id=batch_nodes_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "patent nodes descriptions"}
)
