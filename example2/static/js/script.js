// --- DOM Elements ---
const inputNodesEl = document.getElementById('input-nodes');
const outputNodesEl = document.getElementById('output-nodes');
const outputActivationEl = document.getElementById('output-activation');
const hiddenLayersListEl = document.getElementById('hidden-layers-list');
const addLayerBtn = document.getElementById('add-layer-btn');
const initializeBtn = document.getElementById('initialize-btn');
const configErrorEl = document.getElementById('config-error');

const inputDataEl = document.getElementById('input-data');
const targetDataEl = document.getElementById('target-data');
const learningRateEl = document.getElementById('learning-rate');
const forwardPassBtn = document.getElementById('forward-pass-btn');
const backwardPassBtn = document.getElementById('backward-pass-btn');
const simulationErrorEl = document.getElementById('simulation-error');
const simulationStatusEl = document.getElementById('simulation-status');

const layerOutputsDisplay = document.getElementById('layer-outputs-display');
const layerLogitsDisplay = document.getElementById('layer-logits-display');
const finalOutputDisplay = document.getElementById('final-output-display');
const deltaActivationsDisplay = document.getElementById('delta-activations-display');
const gradientsWDisplay = document.getElementById('gradients-w-display');
const gradientsBDisplay = document.getElementById('gradients-b-display');
const updatedWeightsDisplay = document.getElementById('updated-weights-display');
const updatedBiasesDisplay = document.getElementById('updated-biases-display');

const activationSelectTemplate = document.getElementById('activation-select-template');

const showWeightsBiasesCheckbox = document.getElementById('show-weights-biases');
const showActivationsLogitsCheckbox = document.getElementById('show-activations-logits');
const visualizationContainer = document.getElementById('network-visualization-container');


// --- Global State ---
let currentNetworkStructure = null;
let lastForwardPassResults = null;
let hiddenLayerCount = 1; // Initial hidden layer count

// --- Helper Functions ---

function showError(element, message) {
    element.textContent = message;
    element.style.display = 'block';
}

function clearError(element) {
    element.textContent = '';
    element.style.display = 'none';
}

function showStatus(message) {
    simulationStatusEl.textContent = message;
    simulationStatusEl.style.display = 'block';
    simulationStatusEl.classList.remove('error-message'); // Ensure it's not styled as error
    simulationStatusEl.classList.add('status-message');
}

function showSimulationError(message) {
    simulationStatusEl.textContent = message;
    simulationStatusEl.style.display = 'block';
    simulationStatusEl.classList.remove('status-message');
    simulationStatusEl.classList.add('error-message'); // Style as error
}


function clearStatus() {
    simulationStatusEl.textContent = '';
    simulationStatusEl.style.display = 'none';
}

function clearResultsDisplay() {
    layerOutputsDisplay.textContent = '';
    layerLogitsDisplay.textContent = '';
    finalOutputDisplay.textContent = '';
    deltaActivationsDisplay.textContent = '';
    gradientsWDisplay.textContent = '';
    gradientsBDisplay.textContent = '';
    updatedWeightsDisplay.textContent = '';
    updatedBiasesDisplay.textContent = '';
}

function formatJsonForDisplay(data) {
    if (data === null || data === undefined) return '';
    // Simple formatting for arrays/numbers for better readability in <pre>
    if (Array.isArray(data)) {
        return data.map(item => {
            if (typeof item === 'number') {
                return item.toFixed(4); // Format numbers
            }
            if (Array.isArray(item)) { // Handle nested arrays (like weights)
                return `  [${item.map(subItem => typeof subItem === 'number' ? subItem.toFixed(4) : JSON.stringify(subItem)).join(', ')}]`;
            }
            return JSON.stringify(item);
        }).join('\n');
    }
    return JSON.stringify(data, null, 2); // Fallback for other types
}


// --- Configuration Management ---

function addLayer() {
    hiddenLayerCount++;
    const newLayerDiv = document.createElement('div');
    newLayerDiv.classList.add('layer-config', 'hidden-layer');
    newLayerDiv.setAttribute('data-layer-index', hiddenLayerCount);

    const labelNodes = document.createElement('label');
    labelNodes.textContent = `Layer ${hiddenLayerCount} Nodes:`;
    const inputNodes = document.createElement('input');
    inputNodes.type = 'number';
    inputNodes.classList.add('hidden-nodes');
    inputNodes.value = '3'; // Default value
    inputNodes.min = '1';

    const labelActivation = document.createElement('label');
    labelActivation.textContent = 'Activation:';
    // Clone the select element from the template
    const selectActivation = activationSelectTemplate.content.cloneNode(true).querySelector('select');
    selectActivation.classList.add('hidden-activation'); // Add class for selection later
    selectActivation.value = 'relu'; // Default activation

    const removeBtn = document.createElement('button');
    removeBtn.textContent = 'Remove';
    removeBtn.classList.add('remove-layer-btn');
    removeBtn.onclick = () => removeLayer(removeBtn);

    newLayerDiv.appendChild(labelNodes);
    newLayerDiv.appendChild(inputNodes);
    newLayerDiv.appendChild(labelActivation);
    newLayerDiv.appendChild(selectActivation);
    newLayerDiv.appendChild(removeBtn);

    hiddenLayersListEl.appendChild(newLayerDiv);
    updateLayerIndices(); // Renumber layers after adding
}

function removeLayer(button) {
    const layerToRemove = button.closest('.hidden-layer');
    if (layerToRemove) {
        layerToRemove.remove();
        updateLayerIndices(); // Renumber layers after removing
    }
}

function updateLayerIndices() {
    const hiddenLayers = hiddenLayersListEl.querySelectorAll('.hidden-layer');
    hiddenLayerCount = hiddenLayers.length; // Update global count
    hiddenLayers.forEach((layer, index) => {
        const layerIndex = index + 1;
        layer.setAttribute('data-layer-index', layerIndex);
        layer.querySelector('label:first-of-type').textContent = `Layer ${layerIndex} Nodes:`;
    });
}

function getNetworkConfig() {
    clearError(configErrorEl);
    const layers = [];

    // Input Layer
    const inputNodes = parseInt(inputNodesEl.value, 10);
    if (isNaN(inputNodes) || inputNodes < 1) {
        showError(configErrorEl, "Input layer nodes must be a positive number.");
        return null;
    }
    layers.push({ nodes: inputNodes, activation: null }); // Input layer has no activation in this context

    // Hidden Layers
    const hiddenLayerDivs = hiddenLayersListEl.querySelectorAll('.hidden-layer');
    for (let i = 0; i < hiddenLayerDivs.length; i++) {
        const layerDiv = hiddenLayerDivs[i];
        const nodesEl = layerDiv.querySelector('.hidden-nodes');
        const activationEl = layerDiv.querySelector('.hidden-activation');
        const nodes = parseInt(nodesEl.value, 10);
        const activation = activationEl.value;
        const layerIndex = layerDiv.getAttribute('data-layer-index');

        if (isNaN(nodes) || nodes < 1) {
            showError(configErrorEl, `Hidden layer ${layerIndex} nodes must be a positive number.`);
            return null;
        }
        if (!activation) {
            showError(configErrorEl, `Activation must be selected for hidden layer ${layerIndex}.`);
            return null;
        }
        layers.push({ nodes: nodes, activation: activation });
    }

    // Output Layer
    const outputNodes = parseInt(outputNodesEl.value, 10);
    const outputActivation = outputActivationEl.value;
    if (isNaN(outputNodes) || outputNodes < 1) {
        showError(configErrorEl, "Output layer nodes must be a positive number.");
        return null;
    }
     if (!outputActivation) {
        showError(configErrorEl, `Activation must be selected for the output layer.`);
        return null;
    }
    layers.push({ nodes: outputNodes, activation: outputActivation });

    return { layers: layers };
}

async function initializeNetwork() {
    const config = getNetworkConfig();
    if (!config) return;

    clearStatus();
    clearResultsDisplay(); // Clear previous results
    showStatus("Initializing network...");
    initializeBtn.disabled = true;
    forwardPassBtn.disabled = true;
    backwardPassBtn.disabled = true;

    try {
        const response = await fetch('/api/initialize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `HTTP error! status: ${response.status}`);
        }

        currentNetworkStructure = data;
        lastForwardPassResults = null; // Reset forward pass results
        console.log("Network Initialized:", currentNetworkStructure);
        showStatus("Network initialized successfully.");

        // Call visualization update function (defined in network-visualization.js)
        if (typeof updateVisualization === 'function') {
            updateVisualization(currentNetworkStructure);
            applyVisualizationOptions(); // Apply checkbox states
        } else {
            console.error("updateVisualization function not found.");
        }

        forwardPassBtn.disabled = false; // Enable forward pass
        backwardPassBtn.disabled = true; // Keep backward pass disabled until forward is run

    } catch (error) {
        console.error("Initialization failed:", error);
        showError(configErrorEl, `Initialization failed: ${error.message}`);
        currentNetworkStructure = null;
        // Clear visualization if initialization fails
         if (typeof updateVisualization === 'function') {
            updateVisualization(null); // Or pass an empty structure
        }
    } finally {
        initializeBtn.disabled = false; // Re-enable initialize button
    }
}

// --- Simulation Control ---

function parseCommaSeparatedInput(inputString, fieldName) {
    clearError(simulationErrorEl);
    if (!inputString.trim()) {
         showSimulationError(`Error: ${fieldName} cannot be empty.`);
        return null;
    }
    const parts = inputString.split(',').map(s => s.trim());
    const numbers = parts.map(Number);

    if (numbers.some(isNaN)) {
        showSimulationError(`Error: ${fieldName} must contain only comma-separated numbers.`);
        return null;
    }
    return numbers;
}

async function runForwardPass() {
    if (!currentNetworkStructure) {
        showSimulationError("Error: Network not initialized.");
        return;
    }

    const inputData = parseCommaSeparatedInput(inputDataEl.value, "Input Data");
    if (!inputData) return;

    // Client-side validation of input size
    const expectedInputNodes = currentNetworkStructure.layer_config[0].nodes;
    if (inputData.length !== expectedInputNodes) {
         showSimulationError(`Error: Input data size (${inputData.length}) does not match network input nodes (${expectedInputNodes}).`);
        return;
    }


    clearStatus();
    clearResultsDisplay(); // Clear previous results before new pass
    showStatus("Running forward pass...");
    forwardPassBtn.disabled = true;
    backwardPassBtn.disabled = true;

    try {
        const response = await fetch('/api/forward_pass', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_data: inputData })
        });

        const results = await response.json();

        if (!response.ok) {
            throw new Error(results.error || `HTTP error! status: ${response.status}`);
        }

        lastForwardPassResults = results; // Store results for backward pass
        console.log("Forward Pass Results:", results);
        showStatus("Forward pass completed.");

        // Display results
        layerOutputsDisplay.textContent = formatJsonForDisplay(results.layer_outputs);
        layerLogitsDisplay.textContent = formatJsonForDisplay(results.logits);
        finalOutputDisplay.textContent = formatJsonForDisplay(results.final_output);

        // Update visualization with activation values
        if (typeof updateVisualization === 'function') {
            // Pass activations and logits to visualization
            updateVisualization(currentNetworkStructure, results.layer_outputs, results.logits);
            applyVisualizationOptions();
        }

        backwardPassBtn.disabled = false; // Enable backward pass

    } catch (error) {
        console.error("Forward pass failed:", error);
        showSimulationError(`Forward pass failed: ${error.message}`);
        lastForwardPassResults = null;
    } finally {
        forwardPassBtn.disabled = false; // Re-enable forward pass button
    }
}

async function runBackwardPass() {
    if (!currentNetworkStructure) {
        showSimulationError("Error: Network not initialized.");
        return;
    }
    if (!lastForwardPassResults) {
        showSimulationError("Error: Run forward pass first.");
        return;
    }

    const targetData = parseCommaSeparatedInput(targetDataEl.value, "Target Data");
    if (!targetData) return;

    const learningRate = parseFloat(learningRateEl.value);
     if (isNaN(learningRate) || learningRate <= 0) {
        showSimulationError("Error: Learning rate must be a positive number.");
        return;
    }

    // Client-side validation of target size
    const expectedOutputNodes = currentNetworkStructure.layer_config[currentNetworkStructure.layer_config.length - 1].nodes;
     if (targetData.length !== expectedOutputNodes) {
         showSimulationError(`Error: Target data size (${targetData.length}) does not match network output nodes (${expectedOutputNodes}).`);
        return;
    }


    clearStatus();
    // Don't clear forward pass results, but clear backward results
    deltaActivationsDisplay.textContent = '';
    gradientsWDisplay.textContent = '';
    gradientsBDisplay.textContent = '';
    updatedWeightsDisplay.textContent = '';
    updatedBiasesDisplay.textContent = '';

    showStatus("Running backward pass...");
    backwardPassBtn.disabled = true; // Disable during run
    forwardPassBtn.disabled = true; // Also disable forward during backprop

    try {
        const response = await fetch('/api/backward_pass', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_data: targetData,
                forward_pass_results: lastForwardPassResults, // Send cached results
                learning_rate: learningRate
            })
        });

        const results = await response.json();

        if (!response.ok) {
            throw new Error(results.error || `HTTP error! status: ${response.status}`);
        }

        console.log("Backward Pass Results:", results);
        showStatus("Backward pass completed. Weights and biases updated.");

        // Update network structure with new weights/biases for visualization
        currentNetworkStructure.weights = results.updated_weights;
        currentNetworkStructure.biases = results.updated_biases;

        // Display results
        deltaActivationsDisplay.textContent = formatJsonForDisplay(results.delta_activations);
        gradientsWDisplay.textContent = formatJsonForDisplay(results.gradients_w);
        gradientsBDisplay.textContent = formatJsonForDisplay(results.gradients_b);
        updatedWeightsDisplay.textContent = formatJsonForDisplay(results.updated_weights);
        updatedBiasesDisplay.textContent = formatJsonForDisplay(results.updated_biases);


        // Update visualization with new weights/biases and potentially gradients
        if (typeof updateVisualization === 'function') {
            // Pass updated structure and maybe gradients/deltas for highlighting
            updateVisualization(currentNetworkStructure, lastForwardPassResults.layer_outputs, lastForwardPassResults.logits, results);
            applyVisualizationOptions();
        }

        lastForwardPassResults = null; // Invalidate forward pass results after backprop

    } catch (error) {
        console.error("Backward pass failed:", error);
        showSimulationError(`Backward pass failed: ${error.message}`);
    } finally {
        // Re-enable buttons - User needs to run forward again before next backward
        forwardPassBtn.disabled = false;
        backwardPassBtn.disabled = true; // Must run forward pass again
    }
}

// --- Visualization Options ---
function applyVisualizationOptions() {
    const showWeights = showWeightsBiasesCheckbox.checked;
    const showValues = showActivationsLogitsCheckbox.checked;

    // Add/remove classes or call specific functions in network-visualization.js
    if (typeof setVisualizationOptions === 'function') {
        setVisualizationOptions({ showWeights, showValues });
    } else {
        // Fallback using CSS classes if the function isn't defined
        visualizationContainer.classList.toggle('visualization-hidden-weights', !showWeights);
        visualizationContainer.classList.toggle('visualization-hidden-values', !showValues);
    }
}


// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    addLayerBtn.addEventListener('click', addLayer);
    initializeBtn.addEventListener('click', initializeNetwork);
    forwardPassBtn.addEventListener('click', runForwardPass);
    backwardPassBtn.addEventListener('click', runBackwardPass);

    // Initial setup for the first hidden layer's remove button
    const initialRemoveBtn = hiddenLayersListEl.querySelector('.remove-layer-btn');
    if (initialRemoveBtn) {
        initialRemoveBtn.onclick = () => removeLayer(initialRemoveBtn);
    }

    // Visualization option listeners
    showWeightsBiasesCheckbox.addEventListener('change', applyVisualizationOptions);
    showActivationsLogitsCheckbox.addEventListener('change', applyVisualizationOptions);


    // Initial state
    forwardPassBtn.disabled = true;
    backwardPassBtn.disabled = true;
    clearStatus();
    clearError(configErrorEl);
    clearError(simulationErrorEl);
    clearResultsDisplay();

     // Add default hidden layer if list is empty initially (optional)
    if (hiddenLayersListEl.children.length === 0) {
        // This might conflict if the HTML already includes one layer.
        // Ensure HTML and JS are consistent. If HTML has the first layer,
        // this check might not be needed, or adjust `hiddenLayerCount` init.
        // addLayer(); // Uncomment if you want JS to add the first layer
    } else {
         // Ensure the initial layer count matches the HTML
         hiddenLayerCount = hiddenLayersListEl.querySelectorAll('.hidden-layer').length;
    }

});