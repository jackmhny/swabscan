# /opt/swabscan/ingest_faces.py
import os, glob, cv2, numpy as np, insightface
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

SRC_GLOB   = "/app/input/**/*.jpg"
COLLECTION = "fb_faces"

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_size=(640,640))

client = QdrantClient(url=os.environ.get("QDRANT_URL"))

# (re)create collection
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=rest.VectorParams(size=512, distance=rest.Distance.COSINE),
)

batch = []
BATCH_SIZE = 128
for path in glob.glob(SRC_GLOB, recursive=True):
    img = cv2.imread(path)
    if img is None: continue
    for face in model.get(img):
        vec = face.normed_embedding.astype("float32")
        payload = {"file": path, "bbox": [int(x) for x in face.bbox]}
        batch.append(rest.PointStruct(id=len(batch), vector=vec, payload=payload))
        if len(batch) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION, points=batch)
            batch.clear()
if batch:
    client.upsert(collection_name=COLLECTION, points=batch)
print("Ingestion complete.")

