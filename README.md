# TailorTalk — AI Visual Saree Search
Live Demo: https://tailortalk-image-search7.streamlit.app
A deployable visual-search agent for the TailorTalk assignment.

## What the assignment asks for

- Natural-language chat interface.
- Image upload or image URL.
- Visual similarity search over a saree catalogue.
- Vector index.
- Callable agent/tool interface.
- Streamlit frontend.
- Deployable application + GitHub repository.
- README documenting the design and quality improvements.

The assignment source explicitly requires processing and indexing the supplied saree dataset yourself and emphasizes fine-grained differences such as colour, fabric, weave, print, border and pallu work.

## Architecture

```text
User
  |
  v
Streamlit chat UI
  |
  v
Intent / agent layer
  |
  v
visual_similarity_search tool
  |
  +--> DINOv2 image embedding
  |
  +--> FAISS vector search (candidate retrieval)
  |
  +--> HSV colour histogram reranking
  |
  v
Top-K sarees + scores + thumbnails
```

## Model and vector database

**Embedding model:** `facebook/dinov2-small`

DINOv2 is used because the evaluation is visual and fine-grained rather than primarily text-semantic.

**Vector database:** FAISS using `IndexFlatIP`.

Embeddings are L2-normalized, so inner product is cosine similarity.

**Reranking:** The first-stage FAISS candidates are reranked with a weighted score:

`final = 0.82 * visual_score + 0.18 * color_score`

This makes colour similarity matter when many catalogue images are visually close.

## Dataset

The supplied database is a CSV catalogue with 1,074 rows and an `image_url` column.
The CSV is included locally as `data/catalogue.csv` for the build step.

The project downloads the catalogue images automatically from those URLs.

```bash
python scripts/download_dataset.py
python scripts/build_index.py
```

For a quick test before downloading everything:

```bash
python scripts/download_dataset.py --limit 20
python scripts/build_index.py
```

Do not commit downloaded images to GitHub unless their license explicitly allows redistribution.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Build the vector index:

```bash
python scripts/build_index.py
```

Run the app:

```bash
streamlit run app.py
```

The first run downloads the DINOv2 model from Hugging Face.

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Add the dataset in the deployment environment, or use a permitted hosted dataset/object store.
3. Build the index before deployment if the platform permits the generated artifact, or run indexing as a deployment/startup step.
4. Set the main file to `app.py`.
5. Deploy.

### Hugging Face Spaces

Use the Streamlit SDK, upload the repository, and provide the same model/runtime dependencies.

## Important deployment note

The assignment says the reviewer should need no local setup. Therefore the deployed app must already contain or be able to download the catalogue index and image assets. For a large catalogue, store the images/index in an allowed object store or a Hugging Face dataset and load them during startup.

## Quality improvements

A naive image embedding search can return generic sarees because all catalogue items share the same broad class. This implementation improves retrieval by:

1. Using a visual representation (DINOv2) instead of text embeddings.
2. Retrieving a larger candidate pool before final ranking.
3. Comparing HSV colour distributions.
4. Combining visual and colour scores.
5. Keeping the ranking deterministic and reproducible.
6. Returning the score components so errors are easy to inspect.

## Agent/tool contract

The intended callable tool is:

```text
visual_similarity_search(
    image_path: string,
    top_k: integer = 6
) -> {
    results: [
      {
        path: string,
        score: float,
        visual_score: float,
        color_score: float
      }
    ]
}
```

The UI currently uses a lightweight local intent layer so the demo does not require an LLM API key. If an LLM is added, this tool can be bound to a LangChain agent without changing the search engine.

## Evaluation checklist

Test with at least 10 different catalogue images:

- same dominant colour
- similar border
- similar print
- similar weave/texture
- similar pallu
- mixed colours
- dark vs light sarees

Record Top-1 and Top-5 qualitative relevance and inspect failure cases.

## Trade-offs

- DINOv2 improves visual similarity but adds model download and CPU latency.
- FAISS is simple and free but the index is local to the deployment.
- Colour reranking improves colour-sensitive retrieval but can overemphasize similar colours when patterns differ.
- Exact nearest-neighbour search is accurate for moderate catalogues but uses more memory than approximate indexes.

## Assignment source

This implementation is based on the supplied TailorTalk assignment PDF.
