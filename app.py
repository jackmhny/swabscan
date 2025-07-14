# /opt/swabscan/app.py
import os
import shutil
import cv2
import insightface
import streamlit as st
from PIL import Image
from qdrant_client import QdrantClient
import numpy as np
import zipfile

# ── Config ───────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = "fb_faces"
THUMB_COLS = 12
K_LIMIT    = 144  # max faces per query
INPUT_DIR   = "/app/input"
OUTPUT_DIR  = "/app/output"

# ── Init Qdrant & Face model ─────────────────────────────────────────
@st.cache_resource
def init_models():
    client = QdrantClient(url=QDRANT_URL)
    model  = insightface.app.FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0, det_size=(640, 640))
    return client, model

client, model = init_models()

# ── UI setup ─────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Find a Person")

# ── Sidebar: refs, threshold, select-all ─────────────────────────────
refs = st.sidebar.file_uploader(
    label="Upload 1–2 reference images", type=["jpg","png"], accept_multiple_files=True
)
threshold = st.sidebar.slider("Similarity threshold", 0.0, 1.0, 0.35, 0.01)

if 'select_all' not in st.session_state:
    st.session_state.select_all = False
if st.sidebar.button('Select All'):
    st.session_state.select_all = not st.session_state.select_all

# ── Query Qdrant ─────────────────────────────────────────────────────
points = []
if refs:
    ref_vecs = []
    for ref in refs:
        img = np.array(Image.open(ref).convert("RGB"))
        faces = model.get(img)
        if not faces:
            st.sidebar.warning(f"No face in {ref.name}")
            continue
        ref_vecs.append(faces[0].normed_embedding.astype("float32"))
    if ref_vecs:
        query_vec = np.mean(ref_vecs, axis=0).astype("float32")
        points = client.search(
            collection_name=COLLECTION,
            query_vector=query_vec,
            limit=K_LIMIT,
            with_payload=True,
            score_threshold=threshold
        )
        st.sidebar.success(f"Found {len(points)} candidates")
else:
    st.info("Upload a reference image to search.")
    st.stop()

# ── Display grid & checkboxes ────────────────────────────────────────
cols = st.columns(THUMB_COLS)
selected = []
for idx, pt in enumerate(points):
    p       = pt.payload or {}
    host_fp = p.get("file")
    bbox    = p.get("bbox",[0,0,0,0])
    # map stored path to container mount
    if not host_fp:
        continue
    fname = os.path.basename(host_fp)
    fp    = os.path.join(INPUT_DIR, fname)
    if not os.path.exists(fp):
        continue
    img = cv2.imread(fp)
    h, w = img.shape[:2]
    x0,y0,x1,y1 = (max(0,bbox[0]),max(0,bbox[1]),min(w,bbox[2]),min(h,bbox[3]))
    face = img[y0:y1, x0:x1]
    if face is None or face.size==0: continue
    face  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    thumb = Image.fromarray(face).resize((100,100))
    col   = cols[idx % THUMB_COLS]
    with col:
        st.image(thumb, use_container_width=True)
        key = f"chk_{pt.id}"
        if st.checkbox(label=key, key=key, label_visibility="hidden",
                       value=st.session_state.select_all):
            selected.append((host_fp, fname))

# ── Export & Download ZIP ────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("Export")
st.sidebar.write(f"Selected {len(selected)} photos.")

if st.sidebar.button("Prepare Download"):
    # Ensure output directory exists
    hits_dir = os.path.join(OUTPUT_DIR, "hits_streamlit")
    os.makedirs(hits_dir, exist_ok=True)

    # Copy selected into hits_dir
    for host_fp, fname in selected:
        container_fp = os.path.join(INPUT_DIR, fname)
        shutil.copy2(container_fp, os.path.join(hits_dir, fname))

    # Create ZIP archive
    zip_path = os.path.join(OUTPUT_DIR, "hits_streamlit.zip")
    with zipfile.ZipFile(zip_path, "w") as zp:
        for fn in os.listdir(hits_dir):
            zp.write(os.path.join(hits_dir, fn), arcname=fn)

    # Serve download button
    with open(zip_path, "rb") as f:
        st.sidebar.download_button(
            label="Download ZIP of selected photos",
            data=f,
            file_name="hits_streamlit.zip",
            mime="application/zip"
        )
