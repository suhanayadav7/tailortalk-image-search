from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    image_path: str = Field(description="Local path to the query image.")
    top_k: int = Field(default=6, ge=1, le=20, description="Number of results.")

def make_search_tool(engine):
    def search(image_path: str, top_k: int = 6):
        """Search the saree catalogue for visually similar images."""
        return engine.search(image_path, top_k=top_k)

    return StructuredTool.from_function(
        func=search,
        name="visual_similarity_search",
        description=(
            "Find the closest saree catalogue images to a supplied query image. "
            "Use this tool when the user asks for visually similar, matching, "
            "closest, or look-alike sarees."
        ),
        args_schema=SearchInput,
    )

def run_agent(user_message, image_path, engine, top_k=6):
    text = (user_message or "").lower()
    search_intent = any(
        phrase in text for phrase in [
            "similar", "same", "like this", "find", "match", "look alike",
            "closest", "visual", "saree"
        ]
    )

    if not search_intent:
        return {
            "message": "I can help with visual saree similarity. Ask me to find sarees similar to your image.",
            "results": [],
        }

    tool = make_search_tool(engine)
    results = tool.invoke({"image_path": image_path, "top_k": top_k})
    return {
        "message": f"I found {len(results)} visually similar sarees.",
        "results": results,
    }
