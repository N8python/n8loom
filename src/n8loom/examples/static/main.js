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

function countNonzeroDepths(node) {
    if (!node.children || node.children.length === 0) {
        return 0;
    }
    let nonZeroDepths = 0;
    for (let child of node.children) {
        if (getMaxDepth(child) > 0) {
            nonZeroDepths++;
        }
    }
    return nonZeroDepths;
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
        const branches = countNonzeroDepths(node);
        const span = document.createElement('pre');
        if (index === path.length - 1) {
            span.classList.add('selected-chunk');
        } else {
            span.classList.add('ancestor-chunk');
        }
        if (branches > 1) {
            span.style.backgroundColor = 'blue';
        }
        // Add a space after each node's text except maybe the last
        span.textContent = (node.display_text || 'Empty node');

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

    const buttonGroup = document.createElement('div');
    buttonGroup.className = 'button-group';

    const ramifyBtn = document.createElement('button');
    ramifyBtn.classList.add('ramify-btn');
    ramifyBtn.classList.add('icon-btn');
    ramifyBtn.textContent = '🌱';
    ramifyBtn.onclick = async(e) => {
        e.stopPropagation();
        await ramifySelected();
    }
    if (!selectedNodeData.terminal) {
        buttonGroup.appendChild(ramifyBtn);
    }

    const extendBtn = document.createElement('button');
    extendBtn.classList.add('extend-btn');
    extendBtn.classList.add('icon-btn');
    extendBtn.textContent = '📝';
    extendBtn.onclick = (e) => {
        e.stopPropagation();

        // Create textarea element
        const textarea = document.createElement('textarea');
        textarea.className = 'extend-textarea';
        textarea.value = '';

        // Insert after the last pre element in parentsDiv
        const lastPre = parentsDiv.querySelector('pre:last-of-type');
        lastPre.after(textarea);
        textarea.focus();

        // Auto-resize function
        const adjustHeight = () => {
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        };

        textarea.addEventListener('input', adjustHeight);

        textarea.onkeydown = async(e) => {
            if (e.key === 'Enter' && e.shiftKey) {
                e.preventDefault();
                const text = textarea.value;
                if (text) {
                    const { created_children } = await postJSON(`${BASE_URL}/node/ramify`, {
                        node_id: selectedNodeData.node_id,
                        text: text
                    });
                    selectedNode = created_children[0];

                    // Refresh the tree
                    const treeData = await getJSON(`${BASE_URL}/loom/${currentLoomId}`);
                    updateTree(treeData);
                }
                textarea.remove();
            } else if (e.key === 'Escape') {
                textarea.remove();
            }
        };
    }
    if (!selectedNodeData.terminal) {
        buttonGroup.appendChild(extendBtn);
    }

    // Create a delete button
    const deleteBtn = document.createElement('button');
    deleteBtn.classList.add('delete-btn');
    deleteBtn.classList.add('icon-btn');
    deleteBtn.textContent = '🗑️';
    deleteBtn.onclick = async(e) => {
        e.stopPropagation();
        selectNode(path[path.length - 2].node_id);
        await deleteNode(selectedNodeData.node_id);
    };
    if (path.length > 1) {
        buttonGroup.appendChild(deleteBtn);
    }

    selectedDiv.appendChild(buttonGroup);

    // 3) Children of this node
    const childrenDiv = document.createElement('div');
    childrenDiv.className = 'children-container';

    if (selectedNodeData.children && selectedNodeData.children.length > 0) {
        selectedNodeData.children.forEach(child => {
            const childDiv = document.createElement('div');
            childDiv.className = 'child-node';
            const depth = getMaxDepth(child);
            childDiv.innerHTML = "<pre>" + (child.display_text || 'Empty child') + "</pre>" +
                (depth > 0 ? `<span style="margin: 0 10px;">(...${depth} more levels)</span>` : '');

            // Create a container for the buttons that will be inline
            const buttonContainer = document.createElement('div');
            buttonContainer.style.display = 'inline-flex';
            buttonContainer.style.alignItems = 'center';
            buttonContainer.style.marginLeft = '10px';
            childDiv.appendChild(buttonContainer);

            // Clicking a child sets it as selected
            childDiv.onclick = (e) => {
                e.stopPropagation();
                selectNode(child.node_id);
            };

            // Add a delete button for this child
            const childDeleteBtn = document.createElement('button');
            childDeleteBtn.classList.add('delete-btn');
            childDeleteBtn.classList.add('icon-btn');
            childDeleteBtn.textContent = '🗑️';
            childDeleteBtn.style.width = 'fit-content';
            childDeleteBtn.onclick = async(e) => {
                e.stopPropagation();
                await deleteNode(child.node_id);
            };

            buttonContainer.appendChild(childDeleteBtn);

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
        selectedNode = null;
        currentLoomId = response.loom_id;

        // Fetch and display the initial tree
        const treeData = await getJSON(`${BASE_URL}/loom/${currentLoomId}`);
        updateTree(treeData);

        showStatus(`Loom created successfully: ${currentLoomId}`);
    } catch (error) {
        showStatus(error.message, true);
    }
    refreshLoomList();
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
        const body = {
            node_id: selectedNode,
            stream: true,
            n: parseInt(document.getElementById('numSamples').value, 10),
            temp: parseFloat(document.getElementById('temperature').value),
            max_tokens: parseInt(document.getElementById('maxTokens').value, 10)
        };
        const response = await fetch(`${BASE_URL}/node/ramify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Error ${response.status}: ${errorText}`);
        }
        const childrenDiv = document.querySelector('.children-container');
        // Create side-by-side streaming child nodes for each sample
        const streamChildNodes = [];
        const batchSize = parseInt(document.getElementById('numSamples').value, 10);
        for (let i = 0; i < batchSize; i++) {
            const streamChild = document.createElement('div');
            streamChild.className = 'child-node streaming';
            streamChild.style.pointerEvents = 'none';
            streamChild.innerHTML = "<pre>Generating...</pre>";
            childrenDiv.appendChild(streamChild);
            streamChildNodes.push(streamChild);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let resultText = "";
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            resultText += decoder.decode(value, { stream: true });
            const lines = resultText.split("\n");
            for (let i = 0; i < lines.length - 1; i++) {
                const line = lines[i].trim();
                if (line) {
                    const data = JSON.parse(line);
                    if (data.type === "update") {
                        for (let i = 0; i < data.decoded_texts.length; i++) {
                            streamChildNodes[i].querySelector('pre').textContent = data.decoded_texts[i];
                        }
                    } else if (data.type === "final") {
                        for (let i = 0; i < data.decoded_texts.length; i++) {
                            streamChildNodes[i].querySelector('pre').textContent = data.decoded_texts[i];
                            streamChildNodes[i].style.pointerEvents = 'auto';
                            streamChildNodes[i].classList.remove('streaming');
                            streamChildNodes[i].onclick = () => {
                                selectNode(selectedNode);
                            };
                        }
                    }
                }
            }
            resultText = lines[lines.length - 1];
        }
        const treeData = await getJSON(`${BASE_URL}/loom/${currentLoomId}`);
        updateTree(treeData);
        showStatus('Node ramified successfully (stream)');
    } catch (error) {
        showStatus(error.message, true);
    }
}


// ------------------ New Functions for Loom Management ------------------
async function refreshLoomList() {
    try {
        const looms = await getJSON(`${BASE_URL}/looms`);
        const loomListDiv = document.getElementById('loomList');
        loomListDiv.innerHTML = '';
        looms.forEach(loom => {
            const loomDiv = document.createElement('div');
            loomDiv.style.marginBottom = '10px';

            // Create a span to display only the loom id.
            const loomIdSpan = document.createElement('span');
            loomIdSpan.textContent = loom.loom_id;
            loomIdSpan.style.cursor = 'pointer';
            loomIdSpan.style.marginRight = '10px';
            // Single click loads the loom.
            loomIdSpan.onclick = () => {
                loadLoomById(loom.loom_id);
            };
            // Double-click to edit (inline renaming)
            loomIdSpan.ondblclick = () => {
                const input = document.createElement('input');
                input.type = 'text';
                input.value = loom.loom_id;
                input.style.width = '100px';
                input.onblur = async() => {
                    if (input.value && input.value !== loom.loom_id) {
                        try {
                            await postJSON(`${BASE_URL}/looms/${loom.loom_id}/rename`, { new_id: input.value });
                            showStatus(`Loom renamed to ${input.value}`);
                            refreshLoomList();
                        } catch (error) {
                            showStatus(error.message, true);
                        }
                    }
                    loomIdSpan.textContent = input.value;
                    loomDiv.replaceChild(loomIdSpan, input);
                };
                input.onkeydown = (e) => {
                    if (e.key === 'Enter') {
                        input.blur();
                    }
                };
                loomDiv.replaceChild(input, loomIdSpan);
                input.focus();
            };
            loomDiv.appendChild(loomIdSpan);

            // Create a small export button rendered as an up arrow.
            const exportBtn = document.createElement('button');
            exportBtn.className = 'icon-btn';
            exportBtn.style.padding = '5px 8px';
            exportBtn.textContent = '↑';
            exportBtn.onclick = () => exportLoom(loom.loom_id);
            loomDiv.appendChild(exportBtn);

            // NEW: Create a delete button rendered as a trash icon.
            // NEW: Create a delete button rendered as a trash icon.
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'icon-btn';
            deleteBtn.style.padding = '5px 8px';
            deleteBtn.style.marginLeft = '5px';
            deleteBtn.textContent = '🗑️';
            deleteBtn.onclick = async() => {
                // Use SweetAlert instead of confirm()
                swal({
                    title: "Are you sure?",
                    text: `Are you sure you want to delete loom ${loom.loom_id}?`,
                    icon: "warning",
                    buttons: true,
                    dangerMode: true,
                }).then(async(willDelete) => {
                    if (willDelete) {
                        try {
                            await fetch(`${BASE_URL}/looms/${loom.loom_id}`, { method: 'DELETE' });
                            showStatus(`Loom ${loom.loom_id} deleted successfully`);
                            refreshLoomList();
                        } catch (error) {
                            showStatus(error.message, true);
                        }
                    }
                });
            };
            loomDiv.appendChild(deleteBtn);


            loomListDiv.appendChild(loomDiv);
        });
    } catch (error) {
        showStatus(error.message, true);
    }
}


function findDeepestNode(node, depth = 0) {
    let maxDepth = depth;
    let deepestNode = node;

    if (node.children && node.children.length > 0) {
        for (const child of node.children) {
            const [childNode, childDepth] = findDeepestNode(child, depth + 1);
            if (childDepth > maxDepth) {
                maxDepth = childDepth;
                deepestNode = childNode;
            }
        }
    }

    return [deepestNode, maxDepth];
}

async function loadLoomById(loomId) {
    try {
        const treeData = await getJSON(`${BASE_URL}/loom/${loomId}`);
        currentLoomId = loomId;

        // Find deepest node
        const [deepestNode, _] = findDeepestNode(treeData);
        selectedNode = deepestNode.node_id;

        updateTree(treeData);
        showStatus(`Loaded loom ${loomId}`);
    } catch (error) {
        showStatus(error.message, true);
    }
}

function loadLoomFromInput() {
    const loomId = document.getElementById('loadLoomId').value;
    if (loomId) {
        loadLoomById(loomId);
    }
}

async function exportLoom(loomId) {
    try {
        const exportData = await getJSON(`${BASE_URL}/loom/${loomId}/export`);
        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportData, null, 2));
        const dlAnchorElem = document.createElement('a');
        dlAnchorElem.setAttribute("href", dataStr);
        dlAnchorElem.setAttribute("download", `loom_${loomId}.json`);
        dlAnchorElem.click();
        showStatus(`Loom ${loomId} exported successfully`);
    } catch (error) {
        showStatus(error.message, true);
    }
}

async function importLoomFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async(e) => {
        try {
            const loomData = JSON.parse(e.target.result);
            if (!currentModelId) {
                showStatus('Please load a model first before importing', true);
                return;
            }
            const importResponse = await postJSON(`${BASE_URL}/looms/import`, {
                model_id: currentModelId,
                loom_data: loomData
            });
            showStatus(`Loom imported successfully: ${importResponse.loom_id}`);
            // Optionally refresh the loom list
            refreshLoomList();
        } catch (error) {
            showStatus(error.message, true);
        }
    };
    reader.readAsText(file);
    refreshLoomList();
}

refreshLoomList();
 
// Add 'copy parent texts' utility button in the top right of the visualization div.
function copyParentTexts() {
    const parentsDiv = document.querySelector('.parents-text');
    if (!parentsDiv) {
        showStatus('No parent texts to copy', true);
        return;
    }
    const pres = parentsDiv.querySelectorAll('pre');
    let combinedText = '';
    pres.forEach(pre => {
         combinedText += pre.textContent + '\n';
    });
    navigator.clipboard.writeText(combinedText)
    .then(() => {
         showStatus('Copied parent texts to clipboard');
    })
    .catch(err => {
         showStatus('Failed to copy: ' + err, true);
    });
}

window.addEventListener('load', () => {
    const visualizationDiv = document.querySelector('.visualization');
    if (visualizationDiv) {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'icon-btn';
        copyBtn.style.position = 'absolute';
        copyBtn.style.top = '10px';
        copyBtn.style.right = '10px';
        copyBtn.textContent = '📄';
        copyBtn.title = 'Copy parent texts';
        copyBtn.onclick = copyParentTexts;
        visualizationDiv.appendChild(copyBtn);
    }
});
