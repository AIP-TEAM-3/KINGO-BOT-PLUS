from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import numpy as np

qdrant_client = QdrantClient(url="http://localhost:6333")

qdrant_client.create_collection(
    collection_name="skku_doc",
    vectors_config=VectorParams(size=1024, distance=Distance.DOT),
)

vector_dir = "../data/embeddings"
points = []

for filename in os.listdir(vector_dir):
    doc_id = filename.split(".")[0]

    text = open(os.path.join(f"../data/texts/{doc_id}.txt", filename), "r").read()

    if filename.endswith(".npy"):
        point_id = int(os.path.splitext(filename)[0])
        vector = np.load(os.path.join(vector_dir, filename)).tolist()
        points.append(PointStruct(id=point_id, vector=vector, payload={"text": text}))

operation_info = qdrant_client.upsert(
    collection_name="skku_doc",
    wait=True,
    points=points,
)
