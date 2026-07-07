import requests


payload = {
    "queries": ["Who wrote Hamlet?"],
    "topk": 3,
    "return_scores": True,
}

response = requests.post("http://127.0.0.1:8000/retrieve", json=payload, timeout=60)
response.raise_for_status()
data = response.json()

print("retriever ok")
for idx, item in enumerate(data["result"][0], start=1):
    doc = item["document"]
    score = item.get("score")
    title = doc["contents"].split("\n", 1)[0]
    print(f"{idx}. score={score} title={title}")
