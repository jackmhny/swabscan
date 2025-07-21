import glob, cv2, numpy as np, insightface
import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, PointStruct, Distance

SRC = "input/*"
COLL = "fb_faces"
QDRANT_URL = "http://qdrant:6333"

# 1. init
model  = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_size=(640,640))
client = QdrantClient(url=QDRANT_URL)

# 2. recreate on qdrant 
client.recreate_collection(
    collection_name=COLL,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

# 3. batch upserts
batch, idx = [], 0
for path in glob.glob(SRC, recursive=True):
    img = cv2.imread(path)
    if img is None: continue
    for face in model.get(img):
        vec = face.normed_embedding.astype("float32")
        payload = {"file": os.path.basename(path), "bbox": [int(x) for x in face.bbox]}
        batch.append(PointStruct(id=idx, vector=vec, payload=payload))
        idx += 1
        if len(batch) == 128:
            client.upsert(collection_name=COLL, points=batch)
            batch.clear()
if batch:
    client.upsert(collection_name=COLL, points=batch)

print(f"Ingested {idx} faces → {QDRANT_URL}/{COLL}")

