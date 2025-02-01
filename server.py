from fastapi import FastAPI, Body, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
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
    display_text: str
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
    stream: Optional[bool] = False

class RamifyResponse(BaseModel):
    node_id: str
    created_children: List[str]

class ClipRequest(BaseModel):
    node_id: str
    token_limit: int

class TrimRequest(BaseModel):
    node_id: str
    token_trim: int

# New models for the loom management endpoints
class LoomInfo(BaseModel):
    loom_id: str
    root_heddle_id: str
    prompt: str

class ImportLoomRequest(BaseModel):
    model_id: str
    loom_data: Dict

class ImportLoomResponse(BaseModel):
    loom_id: str
    heddle_id: str

# ------------------------------
# Helper functions
# ------------------------------
def serialize_heddle(node: Heddle, node_id: str) -> NodeInfo:
    # Return basic information about a node
    return NodeInfo(
        node_id=node_id,
        text=node.text,
        display_text=node.display_text(),
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
        "display_text": node.display_text(),
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

# Add this new model near the others:
class RenameLoomRequest(BaseModel):
    new_id: str

# New endpoint to rename a loom.
@app.post("/looms/{loom_id}/rename")
def rename_loom(loom_id: str, req: RenameLoomRequest):
    """
    Rename an existing loom to a new id.
    The new id must not already exist.
    """
    if loom_id not in loom_store:
        raise HTTPException(status_code=404, detail="Loom not found")
    if req.new_id in loom_store:
        raise HTTPException(status_code=400, detail="New loom id already exists")
    # Remove the old entry and reassign the loom under the new id.
    loom = loom_store.pop(loom_id)
    loom_store[req.new_id] = loom
    return {"old_loom_id": loom_id, "new_loom_id": req.new_id}
@app.delete("/looms/{loom_id}")
def delete_loom(loom_id: str):
    """
    Delete a loom from the store, and remove its root node from the heddle store.
    """
    if loom_id not in loom_store:
        raise HTTPException(status_code=404, detail="Loom not found")
    # Remove the loom
    loom = loom_store.pop(loom_id)
    # Also remove its corresponding root node from heddle_store.
    root_heddle_id = None
    for hid, node in list(heddle_store.items()):
        if node is loom:
            root_heddle_id = hid
            del heddle_store[hid]
            break
    return {"deleted_loom_id": loom_id, "deleted_root_heddle_id": root_heddle_id}

@app.get("/node/{node_id}", response_model=NodeInfo)
def get_node_info(node_id: str):
    """
    Get basic info about a node: text, child IDs, terminal status.
    """
    if node_id not in heddle_store:
        raise HTTPException(status_code=404, detail="Node not found")

    node = heddle_store[node_id]
    return serialize_heddle(node, node_id)


@app.post("/node/ramify")
def ramify_node(req: RamifyRequest):
    """
    - If `text` is given as a string, create one text child.
    - If `text` is given as a list of strings, create multiple text children.
    - Otherwise, create children by sampling from the model (n, temp, max_tokens).
      If `stream` is True, stream generation updates.
    """
    if req.node_id not in heddle_store:
        raise HTTPException(status_code=404, detail="Node not found")
    node = heddle_store[req.node_id]
    result = node.ramify(req.text, n=req.n, temp=req.temp, max_tokens=req.max_tokens, stream=req.stream)
    if req.stream:
        import json
        from fastapi.responses import StreamingResponse
        def event_generator():
            for update in result:
                yield json.dumps(update) + "\n"
        return StreamingResponse(event_generator(), media_type="application/json")

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


@app.delete("/node/{node_id}")
def delete_node(node_id: str):
    """
    Delete a node and its children from the store.
    Cannot delete root nodes of looms.
    """
    if node_id not in heddle_store:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Check if this is a root node
    node = heddle_store[node_id]
    for loom in loom_store.values():
        if node is loom:
            raise HTTPException(status_code=400, detail="Cannot delete root node")
    
    # Recursively collect all child node IDs
    def get_child_ids(node):
        children = []
        for child_id, child in heddle_store.items():
            if child.parent is node:
                children.append(child_id)
                children.extend(get_child_ids(child))
        return children
    
    # Delete all children first
    child_ids = get_child_ids(node)
    for child_id in child_ids:
        del heddle_store[child_id]
    
    # Delete the node itself
    del heddle_store[node_id]
    
    return {"node_id": node_id, "deleted_children": child_ids}


@app.get("/node/{node_id}/subtree")
def get_subtree(node_id: str):
    """
    Returns the entire subtree (recursively) from the given node as JSON.
    """
    if node_id not in heddle_store:
        raise HTTPException(status_code=404, detail="Node not found")
    node = heddle_store[node_id]
    return build_subtree_dict(node, node_id)


# ------------------------------
# New endpoints for loom management
# ------------------------------

@app.get("/looms", response_model=List[LoomInfo])
def list_looms():
    """
    Returns a list of all looms currently in memory.
    """
    looms = []
    for loom_id, loom in loom_store.items():
        root_heddle_id = None
        for hid, node in heddle_store.items():
            if node is loom:
                root_heddle_id = hid
                break
        if root_heddle_id:
            looms.append(LoomInfo(loom_id=loom_id, root_heddle_id=root_heddle_id, prompt=loom.text))
    return looms

@app.get("/loom/{loom_id}/export")
def export_loom(loom_id: str):
    """
    Export a given loom (its full tree) as JSON.
    """
    if loom_id not in loom_store:
        raise HTTPException(status_code=404, detail="Loom not found")
    loom_root = loom_store[loom_id]
    root_heddle_id = None
    for hid, node in heddle_store.items():
        if node is loom_root:
            root_heddle_id = hid
            break
    if root_heddle_id is None:
        raise HTTPException(status_code=500, detail="Root node not found in heddle store.")
    exported_tree = build_subtree_dict(loom_root, root_heddle_id)
    return JSONResponse(content=exported_tree)

@app.post("/looms/import", response_model=ImportLoomResponse)
def import_loom(req: ImportLoomRequest):
    """
    Import a loom from an exported JSON structure.
    The client must provide the model_id (which must already be loaded)
    and the loom_data (the exported JSON). The loom will be re-instantiated
    using the provided model/tokenizer and the tree structure rebuilt.
    """
    if req.model_id not in model_store:
        raise HTTPException(status_code=400, detail="Model id not found")
    model_data = model_store[req.model_id]
    model = model_data["model"]
    tokenizer = model_data["tokenizer"]

    loom_json = req.loom_data
    prompt = loom_json.get("text", "Imported Loom")
    new_loom = Loom(model, tokenizer, prompt)
    new_loom_id = get_next_id("loom_id")
    loom_store[new_loom_id] = new_loom
    new_heddle_id = get_next_id("heddle_id")
    heddle_store[new_heddle_id] = new_loom

    def import_subtree(parent, children_list):
        for child in children_list:
            child_text = child.get("text", "")
            # Use the parent's ramify method to add a child with the provided text.
            result = parent.ramify(child_text)
            if result is None:
                continue
            # If multiple children are returned, we take the first.
            if isinstance(result, list):
                result = result[0]
            child_id = get_next_id("heddle_id")
            heddle_store[child_id] = result
            if child.get("children"):
                import_subtree(result, child["children"])
    import_subtree(new_loom, loom_json.get("children", []))
    return ImportLoomResponse(loom_id=new_loom_id, heddle_id=new_heddle_id)

# ------------------------------
# CORS, static files, and root endpoint
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],  # Explicitly list allowed methods
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
