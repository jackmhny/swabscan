# /opt/swabscan/app.py
import os
import shutil
import cv2
import insightface
import streamlit as st
from PIL import Image
from qdrant_client import QdrantClient
import numpy as np

# ── Config ───────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = "fb_faces"
THUMB_COLS = 12
K_LIMIT    = 144  # max faces per query

# ── Init Qdrant & Face model ─────────────────────────────────────────
@st.cache_resource
def init_models():
    client = QdrantClient(url=QDRANT_URL)
    model  = insightface.app.FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0, det_size=(640, 640))
    return client, model

client, model = init_models()

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color:#1E1E1E; color:#E0E0E0; }
  .stSidebar { background-color:#252526; }
  .stButton>button { border:2px solid #4CAF50; border-radius:10px;
    color:#FFF; background:#4CAF50; padding:10px 24px; transition:0.3s; }
  .stButton>button:hover { background:#FFF; color:#4CAF50; }
  .stImage>img { border-radius:10px; box-shadow:0 4px 8px rgba(0,0,0,0.2); }
  .block-container { max-width:100% !important; }
</style>""", unsafe_allow_html=True)

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
    p = pt.payload or {}
    fp = p.get("file"); bbox = p.get("bbox",[0,0,0,0])
    if not fp or not os.path.exists(fp): continue
    img     = cv2.imread(fp); h,w = img.shape[:2]
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
            selected.append(pt)

# ── Export ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("Export")
st.sidebar.write(f"Selected {len(selected)} photos.")
if st.sidebar.button("Export Selected Photos"):
    outdir = "hits_streamlit"
    os.makedirs(outdir, exist_ok=True)
    for pt in selected:
        src = pt.payload["file"]
        shutil.copy2(src, os.path.join(outdir, os.path.basename(src)))
    st.sidebar.success(f"Copied {len(selected)} photos to '{outdir}'")

