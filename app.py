import os
import tempfile
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

from src.search_engine import SimilarityEngine
from src.agent import run_agent

st.set_page_config(page_title="TailorTalk", page_icon="👗", layout="wide")
st.title("👗 TailorTalk")
st.caption("AI-powered visual search for similar sarees")

@st.cache_resource
def load_engine():
    return SimilarityEngine("artifacts/index.faiss", "artifacts/metadata.json")

def get_image_from_url(url):
    r = requests.get(url, timeout=20, headers={"User-Agent": "TailorTalk/1.0"})
    r.raise_for_status()
    from io import BytesIO
    return Image.open(BytesIO(r.content)).convert("RGB")

if not Path("artifacts/index.faiss").exists():
    st.error("The search index is missing. Run the dataset download and indexing commands from the README.")
    st.stop()

engine = load_engine()

with st.sidebar:
    st.header("Search")
    top_k = st.slider("Matches", 3, 12, 6)
    st.write(f"**Catalogue:** {len(engine.metadata)} sarees")
    st.divider()
    st.caption("DINOv2 visual embeddings + FAISS + colour reranking")

uploaded = st.file_uploader("Upload a saree image", type=["jpg", "jpeg", "png", "webp"])
url = st.text_input("Or paste an image URL")

query_image = None
if uploaded:
    query_image = Image.open(uploaded).convert("RGB")
elif url.strip():
    try:
        query_image = get_image_from_url(url.strip())
    except Exception as e:
        st.error(f"Could not load the image: {e}")

if query_image:
    st.image(query_image, caption="Query", width=300)

    prompt = st.chat_input("Ask: Find sarees similar to this image")
    if prompt:
        if not any(x in prompt.lower() for x in ["similar", "match", "find", "closest", "like", "saree"]):
            st.info("Try asking me to find sarees similar to this image.")
        else:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                query_image.save(f.name)
                query_path = f.name
            try:
                with st.spinner("Searching 1,074 catalogue items..."):
                    response = run_agent(prompt, query_path, engine, top_k)
            finally:
                os.unlink(query_path)

            st.subheader(response["message"])
            for start in range(0, len(response["results"]), 4):
                cols = st.columns(4)
                for col, item in zip(cols, response["results"][start:start+4]):
                    with col:
                        st.image(item["path"], use_container_width=True)
                        st.markdown(f"**{item['name'][:70]}**")
                        st.caption(f"SKU: {item['sku']} · Similarity: {item['score']:.3f}")
                        st.caption(
                            f"₹{item['discounted_price']:,.0f} · "
                            f"Visual {item['visual_score']:.3f} · Colour {item['color_score']:.3f}"
                        )
                        if item.get("website_link"):
                            st.link_button("View product", item["website_link"])
else:
    st.info("Upload a saree image or paste an image URL, then ask for similar sarees.")
