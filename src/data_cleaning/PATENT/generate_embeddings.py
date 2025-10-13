import json

def estrai_contenuti_chat_completion(file_path):
    contenuti = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            message = data.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")
            if content:
                contenuti.append(content)
    return contenuti

def crea_file_batch_embeddings(testi, output_file, modello="text-embedding-3-large"):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, testo in enumerate(testi):
            record = {
                "custom_id": f"embedding-{i}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": modello,
                    "input": testo,
                    "encoding_format": "float"
                }
            }
            f.write(json.dumps(record) + "\n")

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Estrai i testi
testi = estrai_contenuti_chat_completion("prova/data2/batch_684c649ada9c81908ee599badf4cae76_output.jsonl")  # o batch_results_nodes.jsonl

# Step 2: Crea file embeddings
crea_file_batch_embeddings(testi, "prova/data2/batch_embeddings_edges.jsonl")

# Step 3: Carica file ed esegui batch
with open("prova/data2/batch_embeddings_edges.jsonl", "rb") as f:
    file_batch = client.files.create(file=f, purpose="batch")

batch = client.batches.create(
    input_file_id=file_batch.id,
    endpoint="/v1/embeddings",
    completion_window="24h",
    metadata={"description": "embedding dei contenuti GPT per edges"}
)

print("Batch embeddings creato con ID:", batch.id)

# Estrai i contenuti dal file originale
testi = estrai_contenuti_chat_completion("prova/data2/batch_684c649cbfc48190978e948a37531db5_output.jsonl")

# Parametri
chunk_size = 3000
current_chunk = 0
import time

while current_chunk * chunk_size < len(testi):
    start = current_chunk * chunk_size
    end = (current_chunk + 1) * chunk_size
    chunk_testi = testi[start:end]

    chunk_file_name = f"prova/data2/batch_embeddings_nodes_{current_chunk}.jsonl"
    crea_file_batch_embeddings(chunk_testi, chunk_file_name)

    with open(chunk_file_name, "rb") as f:
        chunk_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=chunk_file.id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={"description": f"embedding chunk {current_chunk}"}
    )

    print(f"Batch embeddings creato con ID: {batch.id} per chunk {current_chunk}")

    while True:
        batch_info = client.batches.retrieve(batch.id)
        batch_status = batch_info.status
        if batch_status == "completed":
            print(f"✅ Batch {batch.id} completato.")
            break
        elif batch_status == "failed":
            print(f"❌ Batch {batch.id} fallito.")
            break
        else:
            print(f"⏳ Batch {batch.id} ancora in esecuzione. Attesa 60s...")
            time.sleep(60)

    current_chunk += 1