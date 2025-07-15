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
THUMB_COLS = 8
INPUT_DIR  = "/app/input"
OUTPUT_DIR = "/app/output"
GRAHAM_PIC = "graham.png"

@st.cache_resource
def init_models():
    client = QdrantClient(url=QDRANT_URL)
    model  = insightface.app.FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0, det_size=(640, 640))
    return client, model

client, model = init_models()

st.set_page_config(layout="wide")
st.title("Find a Person")

# ── Sidebar: reference + threshold + search ──────────────────────────
with st.sidebar:
    st.markdown("### Reference Image")
    use_graham = st.checkbox("Use Graham's Pic")
    uploads    = st.file_uploader("Or upload your own", type=["jpg","png"], accept_multiple_files=True)
    threshold  = st.slider("Similarity threshold", 0.0, 1.0, 0.35, 0.01)
    search_btn = st.button("Search")
    st.markdown("---")

# Kick off search
if search_btn:
    # Determine refs
    refs = []
    if use_graham and os.path.exists(GRAHAM_PIC):
        refs = [GRAHAM_PIC]
    elif uploads:
        refs = uploads

    if not refs:
        st.sidebar.warning("Please select or upload a reference image.")
        st.stop()

    # Embed refs
    ref_vecs = []
    for ref in refs:
        img = np.array(Image.open(ref).convert("RGB"))
        faces = model.get(img)
        if not faces:
            st.sidebar.warning(f"No face detected in {getattr(ref, 'name', ref)}")
            continue
        ref_vecs.append(faces[0].normed_embedding.astype("float32"))

    if not ref_vecs:
        st.sidebar.error("No valid face embeddings from your references.")
        st.stop()

    # Query Qdrant
    qv = np.mean(ref_vecs, axis=0).astype("float32")
    points = client.search(
        collection_name=COLLECTION,
        query_vector=qv,
        limit=144,
        with_payload=True,
        score_threshold=threshold
    )
    st.sidebar.success(f"Found {len(points)} candidates")

    # Save results in session state for download
    st.session_state.points = points

# Ensure we have prev results
points = st.session_state.get("points", [])

# ── Display grid of all matches ───────────────────────────────────────
cols = st.columns(THUMB_COLS)
for idx, pt in enumerate(points):
    p       = pt.payload or {}
    host_fp = p.get("file"); bbox = p.get("bbox",[0,0,0,0])
    if not host_fp:
        continue

    fname = os.path.basename(host_fp)
    fp    = os.path.join(INPUT_DIR, fname)
    if not os.path.exists(fp):
        continue

    img = cv2.imread(fp); h,w = img.shape[:2]
    x0,y0,x1,y1 = (
        max(0,bbox[0]), max(0,bbox[1]),
        min(w,bbox[2]), min(h,bbox[3])
    )
    if x1<=x0 or y1<=y0:
        continue

    face = img[y0:y1, x0:x1]
    if face is None or face.size==0:
        continue

    face  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    thumb = Image.fromarray(face).resize((200,200))

    col = cols[idx % THUMB_COLS]
    with col:
        st.image(thumb, use_container_width=True)

# ── Download All button ───────────────────────────────────────────────
if points:
    st.sidebar.markdown("---")
    st.sidebar.header("Download All Matches")
    if st.sidebar.button("Prepare ZIP"):
        hits_dir = os.path.join(OUTPUT_DIR, "all_matches")
        if os.path.exists(hits_dir):
            shutil.rmtree(hits_dir)
        os.makedirs(hits_dir, exist_ok=True)
        # copy all faces
        for pt in points:
            host_fp = pt.payload.get("file")
            fname   = os.path.basename(host_fp)
            src     = os.path.join(INPUT_DIR, fname)
            shutil.copy2(src, os.path.join(hits_dir, fname))
        # create zip
        zip_path = os.path.join(OUTPUT_DIR, "all_matches.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        with zipfile.ZipFile(zip_path, "w") as zp:
            for fn in os.listdir(hits_dir):
                zp.write(os.path.join(hits_dir, fn), arcname=fn)
        # serve download
        with open(zip_path, "rb") as f:
            st.sidebar.download_button(
                label="Download ZIP of All Matches",
                data=f,
                file_name="all_matches.zip",
                mime="application/zip"
            )

