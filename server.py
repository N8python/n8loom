
from fastapi import FastAPI, Body, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import uvicorn
from loom import Heddle, Loom
from mlx_lm import load
app = FastAPI()

# For storing loaded models and created nodes.
# In a real application, consider a proper database or other persistent store.
model_store: Dict[str, Dict] = {}     # Map of model_path -> {"model": model, "tokenizer": tokenizer}
loom_store: Dict[str, Loom] = {}      # Map of loom_id -> Loom (the root heddle)
heddle_store: Dict[str, Heddle] = {}  # All nodes (including loom roots), keyed by a unique ID

# Simple integer counters for unique IDs
COUNTERS = {
    "loom_id": 0,
    "heddle_id": 0
}

def get_next_id(prefix: str) -> str:
    COUNTERS[prefix] += 1
    return f"{prefix}-{COUNTERS[prefix]}"

# ------------------------------
# Pydantic models for request/response schemas
# ------------------------------
class LoadModelRequest(BaseModel):
    model_path: str

class LoadModelResponse(BaseModel):
    model_id: str

class CreateLoomRequest(BaseModel):
    model_id: str
    prompt: str

class CreateLoomResponse(BaseModel):
    loom_id: str
    heddle_id: str  # same as the loom's root node ID

class NodeInfo(BaseModel):
    node_id: str
    text: str
    children_ids: List[str]
    terminal: bool

class RamifyRequest(BaseModel):
    node_id: str
    # Provide either "text" or generation parameters
    text: Optional[Union[str, List[str]]] = None

    # generation parameters if we are sampling from the model
    n: Optional[int] = 4
    temp: Optional[float] = 0.8
    max_tokens: Optional[int] = 8

class RamifyResponse(BaseModel):
    node_id: str
    created_children: List[str]

class ClipRequest(BaseModel):
    node_id: str
    token_limit: int

class TrimRequest(BaseModel):
    node_id: str
    token_trim: int

# ------------------------------
# Helper functions
# ------------------------------
def serialize_heddle(node: Heddle, node_id: str) -> NodeInfo:
    # Return basic information about a node
    return NodeInfo(
        node_id=node_id,
        text=node.text,
        children_ids=[
            _id for _id, h in heddle_store.items() if h.parent is node
        ],
        terminal=node.terminal
    )

def build_subtree_dict(node: Heddle, node_id: str) -> Dict:
    """Recursively build a JSON-serializable dict describing the subtree."""
    return {
        "node_id": node_id,
        "text": node.text,
        "terminal": node.terminal,
        "children": [
            build_subtree_dict(child, _id)
            for _id, child in heddle_store.items() if child.parent is node
        ]
    }

# ------------------------------
# API Endpoints
# ------------------------------

@app.post("/load_model", response_model=LoadModelResponse)
def load_model(req: LoadModelRequest):
    """
    Load a model + tokenizer using `mlx_lm.load` and store them under the model_path.
    If the model is already loaded, return the existing path.
    """
    if req.model_path not in model_store:
        # load model only if not already loaded
        model, tokenizer = load(req.model_path)
        model_store[req.model_path] = {
            "model": model,
            "tokenizer": tokenizer
        }
    return LoadModelResponse(model_id=req.model_path)


@app.post("/create_loom", response_model=CreateLoomResponse)
def create_loom(req: CreateLoomRequest):
    """
    Create a new Loom with the given model_path and user prompt.
    """
    if req.model_id not in model_store:
        raise HTTPException(status_code=400, detail="Model path not found")

    model_data = model_store[req.model_id]
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    loom_id = get_next_id("loom_id")
    root_loom = Loom(model, tokenizer, req.prompt)

    # Store the loom in memory
    loom_store[loom_id] = root_loom

    # Also store it in the heddle store
    heddle_id = get_next_id("heddle_id")
    heddle_store[heddle_id] = root_loom

    return CreateLoomResponse(loom_id=loom_id, heddle_id=heddle_id)


@app.get("/loom/{loom_id}")
def get_loom_info(loom_id: str):
    """
    Returns a JSON subtree of the entire Loom structure.
    """
    if loom_id not in loom_store:
        raise HTTPException(status_code=404, detail="Loom not found")

    loom_root = loom_store[loom_id]
    # We need to find which heddle_id references loom_root
    root_heddle_id = None
    for hid, node in heddle_store.items():
        if node is loom_root:
            root_heddle_id = hid
            break

    if root_heddle_id is None:
        raise HTTPException(status_code=500, detail="Root node not found in heddle store.")

    return build_subtree_dict(loom_root, root_heddle_id)


@app.get("/node/{node_id}", response_model=NodeInfo)
def get_node_info(node_id: str):
    """
    Get basic info about a node: text, child IDs, terminal status.
    """
    if node_id not in heddle_store:
        raise HTTPException(status_code=404, detail="Node not found")

    node = heddle_store[node_id]
    return serialize_heddle(node, node_id)


@app.post("/node/ramify", response_model=RamifyResponse)
def ramify_node(req: RamifyRequest):
    """
    - If `text` is given as a string, create one text child.
    - If `text` is given as a list of strings, create multiple text children.
    - Otherwise, create children by sampling from the model (n, temp, max_tokens).
    """
    if req.node_id not in heddle_store:
        raise HTTPException(status_code=404, detail="Node not found")

    node = heddle_store[req.node_id]
    result = node.ramify(req.text, n=req.n, temp=req.temp, max_tokens=req.max_tokens)

    created_children_ids = []

    if isinstance(result, Heddle):
        # Single child created
        child_id = get_next_id("heddle_id")
        heddle_store[child_id] = result
        created_children_ids.append(child_id)
    elif isinstance(result, list) and all(isinstance(r, Heddle) for r in result):
        # Multiple children created
        for child in result:
            child_id = get_next_id("heddle_id")
            heddle_store[child_id] = child
            created_children_ids.append(child_id)
    else:
        # Possibly no children created (if node was terminal)
        pass

    return RamifyResponse(node_id=req.node_id, created_children=created_children_ids)


@app.post("/node/clip")
def clip_node(req: ClipRequest):
    """
    Clip the node (and remove its children) to `token_limit`.
    """
    if req.node_id not in heddle_store:
        raise HTTPException(status_code=404, detail="Node not found")
    node = heddle_store[req.node_id]
    node.clip(req.token_limit)
    return {"node_id": req.node_id, "clipped_to": req.token_limit}


@app.post("/node/trim")
def trim_node(req: TrimRequest):
    """
    Trim the last N tokens from the node (and remove its children).
    """
    if req.node_id not in heddle_store:
        raise HTTPException(status_code=404, detail="Node not found")
    node = heddle_store[req.node_id]
    node.trim(req.token_trim)
    return {"node_id": req.node_id, "trimmed_tokens": req.token_trim}


@app.get("/node/{node_id}/subtree")
def get_subtree(node_id: str):
    """
    Returns the entire subtree (recursively) from the given node as JSON.
    """
    if node_id not in heddle_store:
        raise HTTPException(status_code=404, detail="Node not found")
    node = heddle_store[node_id]
    return build_subtree_dict(node, node_id)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return Response(
        content="""
        <html>
          <head>
            <meta http-equiv="refresh" content="0; url=/static/index.html" />
          </head>
          <body>
            <p>Redirecting to the client...</p>
          </body>
        </html>
        """,
        media_type="text/html",
    )
# ------------------------------
# Run the server (for local testing)
# ------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
