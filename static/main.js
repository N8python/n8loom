const BASE_URL = 'http://localhost:8000';
let selectedNode = null;
let currentModelId = null;
let currentLoomId = null;
let loomData = null; // Will store the entire tree structure once fetched

// Helper functions for API calls
async function postJSON(url, data) {
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Error ${response.status}: ${errorText}`);
    }

    return await response.json();
}

async function getJSON(url) {
    const response = await fetch(url);
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Error ${response.status}: ${errorText}`);
    }
    return await response.json();
}

function showStatus(message, isError = false) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = 'status ' + (isError ? 'error' : 'success');
}

// ------------------ Tree Rendering Logic ------------------

// Recursively search for node + path from root to nodeId
// Calculate the maximum depth below a node
function getMaxDepth(node) {
    if (!node.children || node.children.length === 0) {
        return 0;
    }
    return 1 + Math.max(...node.children.map(child => getMaxDepth(child)));
}

function findNodeAndPath(root, targetId, path = []) {
    // If this is the node
    if (root.node_id === targetId) {
        return [...path, root];
    }

    // If children exist, search them
    if (root.children) {
        for (let child of root.children) {
            const result = findNodeAndPath(child, targetId, [...path, root]);
            if (result) {
                return result;
            }
        }
    }
    return null; // Not found in this branch
}

/**
 * Renders the entire "relative" view inside #tree:
 *  1) All ancestor texts (including selected) concatenated at the top
 *     - Each ancestor's text is in its own clickable <span>.
 *  2) The selected node in the center
 *  3) The immediate children below
 */
function renderRelativeView(selectedId) {
    const treeContainer = document.getElementById('tree');
    treeContainer.innerHTML = '';

    // If no data or no selected node, just return
    if (!loomData || !selectedId) {
        return;
    }

    const path = findNodeAndPath(loomData, selectedId);
    if (!path) {
        treeContainer.textContent = 'Selected node not found.';
        return;
    }

    // 1) Build a parent text container
    const parentsDiv = document.createElement('div');
    parentsDiv.className = 'parents-text';

    // For each ancestor node, create a span that:
    //    - shows node.text
    //    - highlights on hover
    //    - on click, re-selects that ancestor
    path.forEach((node, index) => {
        const span = document.createElement('span');
        if (index === path.length - 1) {
            span.classList.add('selected-chunk');
        } else {
            span.classList.add('ancestor-chunk');
        }
        // Add a space after each node's text except maybe the last
        span.textContent = (node.text || 'Empty node') + ' ';

        // Clicking on this chunk re-selects that node
        span.onclick = (e) => {
            e.stopPropagation();
            selectNode(node.node_id);
        };
        parentsDiv.appendChild(span);
    });

    // Add the parentsDiv to the DOM
    treeContainer.appendChild(parentsDiv);

    // 2) The selected node (last in path)
    const selectedNodeData = path[path.length - 1];

    const selectedDiv = document.createElement('div');
    selectedDiv.className = 'selected-node';


    // Create a delete button
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-btn';
    deleteBtn.textContent = '🗑️';
    deleteBtn.onclick = async(e) => {
        e.stopPropagation();
        selectNode(path[path.length - 2].node_id);
        await deleteNode(selectedNodeData.node_id);
    };
    if (path.length > 1) {
        selectedDiv.appendChild(deleteBtn);
    }

    const ramifyBtn = document.createElement('button');
    ramifyBtn.className = 'ramify-btn';
    ramifyBtn.textContent = '🌱';
    ramifyBtn.onclick = async(e) => {
        e.stopPropagation();
        await ramifySelected();
    }
    selectedDiv.appendChild(ramifyBtn);

    // 3) Children of this node
    const childrenDiv = document.createElement('div');
    childrenDiv.className = 'children-container';

    if (selectedNodeData.children && selectedNodeData.children.length > 0) {
        selectedNodeData.children.forEach(child => {
            const childDiv = document.createElement('div');
            childDiv.className = 'child-node';
            const depth = getMaxDepth(child);
            childDiv.textContent = (child.text || 'Empty child') + (depth > 0 ? ` (...${depth} more levels)` : '');

            // Clicking a child sets it as selected
            childDiv.onclick = (e) => {
                e.stopPropagation();
                selectNode(child.node_id);
            };

            // Add a delete button for this child
            const childDeleteBtn = document.createElement('button');
            childDeleteBtn.className = 'delete-btn';
            childDeleteBtn.textContent = '🗑️';
            childDeleteBtn.style.width = 'fit-content';
            childDeleteBtn.onclick = async(e) => {
                e.stopPropagation();
                await deleteNode(child.node_id);
            };

            childDiv.appendChild(childDeleteBtn);

            childrenDiv.appendChild(childDiv);
        });
    }

    parentsDiv.appendChild(selectedDiv);
    treeContainer.appendChild(childrenDiv);
}

/**
 * Updates the global loomData and triggers rendering of the relative view.
 * If there is no selected node, we set it to the root node (loomData.node_id).
 */
function updateTree(treeData) {
    loomData = treeData;

    // If there is no current selection, default to the root node
    if (!selectedNode) {
        selectedNode = loomData.node_id;
    }

    renderRelativeView(selectedNode);
}

/**
 * Sets the selected node and re-renders the relative view.
 */
function selectNode(nodeId) {
    selectedNode = nodeId;
    renderRelativeView(selectedNode);
}


// ------------------ API Wrappers ------------------

async function loadModel() {
    try {
        const modelPath = document.getElementById('modelPath').value;
        const response = await postJSON(`${BASE_URL}/load_model`, {
            model_path: modelPath
        });
        currentModelId = response.model_id;
        showStatus(`Model loaded successfully: ${currentModelId}`);
    } catch (error) {
        showStatus(error.message, true);
    }
}

async function createLoom() {
    if (!currentModelId) {
        showStatus('Please load a model first', true);
        return;
    }

    try {
        const prompt = document.getElementById('prompt').value;
        const response = await postJSON(`${BASE_URL}/create_loom`, {
            model_id: currentModelId,
            prompt: prompt
        });

        currentLoomId = response.loom_id;

        // Fetch and display the initial tree
        const treeData = await getJSON(`${BASE_URL}/loom/${currentLoomId}`);
        updateTree(treeData);

        showStatus(`Loom created successfully: ${currentLoomId}`);
    } catch (error) {
        showStatus(error.message, true);
    }
}

async function deleteNode(nodeId) {
    try {
        await fetch(`${BASE_URL}/node/${nodeId}`, {
            method: 'DELETE'
        });

        // Refresh the tree
        const treeData = await getJSON(`${BASE_URL}/loom/${currentLoomId}`);
        updateTree(treeData);

        // If we deleted the currently selected node, reset selection to root
        if (selectedNode === nodeId) {
            selectedNode = loomData.node_id;
            renderRelativeView(selectedNode);
        }

        showStatus('Node deleted successfully');
    } catch (error) {
        showStatus(error.message, true);
    }
}

async function ramifySelected() {
    if (!selectedNode) {
        showStatus('Please select a node first', true);
        return;
    }

    try {
        const ramifyText = document.getElementById('ramifyText').value;
        const body = { node_id: selectedNode };

        if (ramifyText) {
            // If user provided custom text
            body.text = ramifyText;
        } else {
            // Otherwise generate new text from the model
            body.n = parseInt(document.getElementById('numSamples').value, 10);
            body.temp = parseFloat(document.getElementById('temperature').value);
            body.max_tokens = parseInt(document.getElementById('maxTokens').value, 10);
        }

        await postJSON(`${BASE_URL}/node/ramify`, body);

        // Refresh the tree
        const treeData = await getJSON(`${BASE_URL}/loom/${currentLoomId}`);
        updateTree(treeData);

        showStatus('Node ramified successfully');
    } catch (error) {
        showStatus(error.message, true);
    }
}