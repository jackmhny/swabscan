import os
import io
import cv2
import zipfile
import insightface
import numpy as np
import streamlit as st
from PIL import Image
from qdrant_client import QdrantClient

# ── Config ───────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
COLLECTION = "fb_faces"
THUMB_COLS = 4 # Fewer columns = bigger images
PADDING_FACTOR = 0.5 # Add 50% padding around the face bbox
INPUT_DIR  = "/app/input"
GRAHAM_PIC = "graham.png"
GRAHAM_PIC2 = "graham2.jpg"

# ── Model & Client Initialization ────────────────────────────────────
@st.cache_resource
def init_models():
    """Initializes and caches the Qdrant client and FaceAnalysis model."""
    client = QdrantClient(url=QDRANT_URL)
    model  = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=['CPUExecutionProvider']
    )
    model.prepare(ctx_id=0, det_size=(640, 640))
    return client, model

client, model = init_models()

# ── Helper Functions ─────────────────────────────────────────────────
def create_zip_in_memory(points):
    """
    Creates a ZIP file in memory containing the full source images of all matches.
    This avoids writing to disk and combines preparation and download into one step.
    """
    zip_buffer = io.BytesIO()
    # Use a set to avoid adding duplicate files to the zip
    added_files = set()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zp:
        for pt in points:
            host_fp = pt.payload.get("file")
            if not host_fp:
                continue
            
            fname = os.path.basename(host_fp)
            # Check if file has already been added
            if fname in added_files:
                continue

            src_path = os.path.join(INPUT_DIR, fname)
            if os.path.exists(src_path):
                zp.write(src_path, arcname=fname)
                added_files.add(fname)
                
    # Reset buffer position to the beginning before reading
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ── Streamlit App UI ─────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("Find a Swab")
# Button to download the full face archive (served separately via Caddy as /pduddy.tar.gz)
# st.markdown(
#     '<a href="/pduddy.tar.gz" target="_blank">'
#     '<button>Download Full Archive</button></a>',
#     unsafe_allow_html=True,
# )

# Check for query params to trigger an auto-search
params = st.query_params
search_on_load = params.get("search") == "graham"
expander_state = not search_on_load

# Controls are moved out of the sidebar and into an expander for better mobile UI
with st.expander("Search Controls", expanded=expander_state):
    st.markdown("### Reference Image")
    use_graham = st.checkbox("Use Graham's Pic", value=search_on_load)
    uploads    = st.file_uploader("Or upload your own", type=["jpg", "png"], accept_multiple_files=True)
    
    st.markdown("### Search Parameters")
    threshold  = st.slider("Similarity threshold", 0.0, 1.0, 0.35, 0.01)
    search_btn = st.button("Search", use_container_width=True)

# ── Search Logic ─────────────────────────────────────────────────────
if search_btn or search_on_load:
    # Determine reference images
    refs = []
    if uploads and not search_on_load:
        refs = uploads
    elif use_graham or search_on_load:
        if os.path.exists(GRAHAM_PIC):
            refs.append(GRAHAM_PIC)
        if os.path.exists(GRAHAM_PIC2):
            refs.append(GRAHAM_PIC2)

    if not refs:
        st.warning("Please select or upload a reference image.")
        st.stop()
    
    with st.spinner("Embedding references and searching for matches..."):
        # Embed reference faces
        ref_vecs = []
        for ref in refs:
            try:
                img = np.array(Image.open(ref).convert("RGB"))
                faces = model.get(img)
                if not faces:
                    st.warning(f"No face detected in {getattr(ref, 'name', ref)}")
                    continue
                # Use the first detected face's embedding
                ref_vecs.append(faces[0].normed_embedding.astype("float32"))
            except Exception as e:
                st.error(f"Could not process {getattr(ref, 'name', ref)}: {e}")

        if not ref_vecs:
            st.error("Could not generate a valid face embedding from your reference images.")
            st.stop()

        # Average the embeddings if multiple references are provided
        query_vector = np.mean(ref_vecs, axis=0).astype("float32")

        response = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=144,
            with_payload=True,
            score_threshold=threshold
        )

        # Extract the list of ScoredPoint objects from the response
        points = response.points
        
        # Store the actual list of points in the session state
        st.session_state.points = points
        
        # Now len(points) will work correctly on the list
        st.success(f"🎉 Found {len(points)} potential matches!")

        # Clear query params after auto-search to prevent re-triggering
        if search_on_load:
            st.query_params.clear()


# ── Results Display ───────────────────────────────────────────────────
# Retrieve points from session state if they exist
points = st.session_state.get("points", [])

if points:
    # --- Download Button ---
    # Placed here for immediate visibility after a successful search.
    # It now creates the zip in memory on the fly.
    zip_data = create_zip_in_memory(points)
    st.download_button(
        label="Download All Matches as ZIP",
        data=zip_data,
        file_name="all_matches.zip",
        mime="application/zip",
        use_container_width=True
    )
    
    st.markdown("---")

    # --- Image Grid ---
    cols = st.columns(THUMB_COLS)
    for idx, pt in enumerate(points):
        payload = pt.payload or {}
        host_fp = payload.get("file")
        bbox = payload.get("bbox", [0, 0, 0, 0])
        
        if not host_fp:
            continue

        full_path = os.path.join(INPUT_DIR, os.path.basename(host_fp))
        if not os.path.exists(full_path):
            continue

        try:
            img = cv2.imread(full_path)
            h, w = img.shape[:2]

            # Calculate padding to add context around the face
            x0, y0, x1, y1 = bbox
            face_w, face_h = x1 - x0, y1 - y0
            pad_x = int(face_w * PADDING_FACTOR)
            pad_y = int(face_h * PADDING_FACTOR)

            # Apply padding, ensuring coordinates are within image bounds
            padded_x0 = max(0, x0 - pad_x)
            padded_y0 = max(0, y0 - pad_y)
            padded_x1 = min(w, x1 + pad_x)
            padded_y1 = min(h, y1 + pad_y)

            # Crop the face with padding
            face_crop = img[padded_y0:padded_y1, padded_x0:padded_x1]
            
            if face_crop.size == 0:
                continue
            
            # Convert color space for displaying with PIL/Streamlit
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            thumb = Image.fromarray(face_rgb).resize((200, 200))

            # Display in the grid
            col = cols[idx % THUMB_COLS]
            with col:
                st.image(thumb, caption=f"Score: {pt.score:.2f}", use_container_width=True)
        
        except Exception as e:
            # Silently skip images that fail to process to avoid crashing the app
            # You could log this error for debugging if needed
            # st.error(f"Failed to process {os.path.basename(host_fp)}")
            continue
