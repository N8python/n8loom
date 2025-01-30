const BASE_URL = 'http://localhost:8000';
let selectedNode = null;
let currentModelId = null;
let currentLoomId = null;

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

// UI helper functions
function showStatus(message, isError = false) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = `status ${isError ? 'error' : 'success'}`;
}

function createNodeElement(node) {
    const nodeDiv = document.createElement('div');
    nodeDiv.className = 'node';
    nodeDiv.id = node.node_id;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'node-content';

    const textDiv = document.createElement('div');
    textDiv.className = 'node-text';
    textDiv.textContent = node.text || 'Empty node';

    const actionsDiv = document.createElement('div');
    actionsDiv.className = 'node-actions';

    // Add delete button
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '🗑️';
    deleteBtn.className = 'delete-btn';
    deleteBtn.onclick = async(e) => {
        e.stopPropagation();
        await deleteNode(node.node_id);
    };
    actionsDiv.appendChild(deleteBtn);

    nodeDiv.onclick = (e) => {
        e.stopPropagation();
        selectNode(node.node_id);
    };

    contentDiv.appendChild(textDiv);
    contentDiv.appendChild(actionsDiv);
    nodeDiv.appendChild(contentDiv);

    if (node.children && node.children.length > 0) {
        const childrenDiv = document.createElement('div');
        childrenDiv.className = 'node-children';
        node.children.forEach(child => {
            childrenDiv.appendChild(createNodeElement(child));
        });
        nodeDiv.appendChild(childrenDiv);
    }

    return nodeDiv;
}

function updateTree(treeData) {
    const treeContainer = document.getElementById('tree');
    treeContainer.innerHTML = '';
    treeContainer.appendChild(createNodeElement(treeData));
}

function selectNode(nodeId) {
    // Clear previous selection
    const previousSelected = document.querySelector('.node.selected');
    if (previousSelected) {
        previousSelected.classList.remove('selected');
    }

    // Set new selection
    const nodeElement = document.getElementById(nodeId);
    if (nodeElement) {
        nodeElement.classList.add('selected');
        selectedNode = nodeId;
        document.getElementById('ramifyBtn').disabled = false;
    }
}

// Main functionality
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

        // Refresh the tree view
        const treeData = await getJSON(`${BASE_URL}/loom/${currentLoomId}`);
        updateTree(treeData);

        // Clear selection if deleted node was selected
        if (selectedNode === nodeId) {
            selectedNode = null;
            document.getElementById('ramifyBtn').disabled = true;
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
        const body = {
            node_id: selectedNode
        };

        if (ramifyText) {
            body.text = ramifyText;
        } else {
            body.n = parseInt(document.getElementById('numSamples').value);
            body.temp = parseFloat(document.getElementById('temperature').value);
            body.max_tokens = parseInt(document.getElementById('maxTokens').value);
        }

        await postJSON(`${BASE_URL}/node/ramify`, body);

        // Refresh the tree view
        const treeData = await getJSON(`${BASE_URL}/loom/${currentLoomId}`);
        updateTree(treeData);

        showStatus('Node ramified successfully');
    } catch (error) {
        showStatus(error.message, true);
    }
}

// Initialize the UI
document.addEventListener('DOMContentLoaded', () => {
    // Disable ramify button initially
    document.getElementById('ramifyBtn').disabled = true;
});