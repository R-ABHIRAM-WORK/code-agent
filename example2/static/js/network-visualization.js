// --- D3 Setup ---
const svg = d3.select("#network-svg");
const svgContainer = document.getElementById('network-visualization-container');

// Tooltip div
const tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

// Visualization options state
let vizOptions = {
    showWeights: true,
    showValues: true
};

// --- Color Mapping ---
const activationColorMapping = {
    'relu': 'activation-relu',
    'gelu': 'activation-gelu',
    'sigmoid': 'activation-sigmoid',
    'tanh': 'activation-tanh',
    'softmax': 'activation-softmax',
    'linear': 'activation-linear',
    // Add more if needed
};

// --- Helper Functions ---

function getLayerType(layerIndex, totalLayers) {
    if (layerIndex === 0) return 'input-layer';
    if (layerIndex === totalLayers - 1) return 'output-layer';
    return 'hidden-layer';
}

function formatValue(value) {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') return value.toFixed(3);
    return String(value);
}

// --- Main Drawing Function ---

function updateVisualization(networkStructure, layerOutputs = null, layerLogits = null, backwardResults = null) {
    svg.selectAll("*").remove(); // Clear previous visualization

    if (!networkStructure || !networkStructure.layer_config || networkStructure.layer_config.length === 0) {
        svg.append("text")
           .attr("x", "50%")
           .attr("y", "50%")
           .attr("text-anchor", "middle")
           .text("Network not initialized.");
        return;
    }

    const layerConfig = networkStructure.layer_config;
    const weights = networkStructure.weights || [];
    const biases = networkStructure.biases || [];
    const numLayers = layerConfig.length;

    // --- Layout Calculation ---
    const containerWidth = svgContainer.clientWidth;
    const containerHeight = svgContainer.clientHeight;
    const margin = { top: 40, right: 60, bottom: 40, left: 60 };
    const width = containerWidth - margin.left - margin.right;
    const height = containerHeight - margin.top - margin.bottom;

    const layerGap = width / (numLayers > 1 ? numLayers - 1 : 1); // Horizontal gap between layers
    const maxNodesInLayer = Math.max(...layerConfig.map(l => l.nodes));
    const nodeRadius = Math.min(15, height / (maxNodesInLayer * 2.5)); // Adjust radius based on height and max nodes

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    // --- Data Preparation (Nodes and Links) ---
    const nodes = [];
    const links = [];

    layerConfig.forEach((layer, layerIndex) => {
        const layerNodes = layer.nodes;
        const layerX = layerIndex * layerGap;
        const nodeGap = height / (layerNodes + 1); // Vertical gap between nodes in a layer

        for (let nodeIndex = 0; nodeIndex < layerNodes; nodeIndex++) {
            const nodeY = nodeGap * (nodeIndex + 1);
            const nodeId = `l${layerIndex}-n${nodeIndex}`;
            const activation = layer.activation; // Activation function for this layer
            const nodeData = {
                id: nodeId,
                layerIndex: layerIndex,
                nodeIndex: nodeIndex,
                x: layerX,
                y: nodeY,
                activation: activation,
                layerType: getLayerType(layerIndex, numLayers),
                bias: (layerIndex > 0 && biases[layerIndex - 1]) ? biases[layerIndex - 1][nodeIndex] : null,
                // Values from forward pass
                activationValue: (layerOutputs && layerOutputs[layerIndex]) ? layerOutputs[layerIndex][nodeIndex] : null,
                logitValue: (layerIndex > 0 && layerLogits && layerLogits[layerIndex - 1]) ? layerLogits[layerIndex - 1][nodeIndex] : null,
                // Gradients from backward pass (optional)
                deltaActivation: (backwardResults && backwardResults.delta_activations && backwardResults.delta_activations[layerIndex]) ? backwardResults.delta_activations[layerIndex][nodeIndex] : null,
                biasGradient: (backwardResults && backwardResults.gradients_b && backwardResults.gradients_b[layerIndex - 1]) ? backwardResults.gradients_b[layerIndex - 1][nodeIndex] : null,
            };
            nodes.push(nodeData);

            // Create links from previous layer to this node
            if (layerIndex > 0) {
                const prevLayerNodes = layerConfig[layerIndex - 1].nodes;
                for (let prevNodeIndex = 0; prevNodeIndex < prevLayerNodes; prevNodeIndex++) {
                    const prevNodeId = `l${layerIndex - 1}-n${prevNodeIndex}`;
                    const weight = (weights[layerIndex - 1] && weights[layerIndex - 1][nodeIndex]) ? weights[layerIndex - 1][nodeIndex][prevNodeIndex] : null;
                    const weightGradient = (backwardResults && backwardResults.gradients_w && backwardResults.gradients_w[layerIndex - 1] && backwardResults.gradients_w[layerIndex - 1][nodeIndex]) ? backwardResults.gradients_w[layerIndex - 1][nodeIndex][prevNodeIndex] : null;

                    links.push({
                        id: `link-${prevNodeId}-to-${nodeId}`,
                        source: nodes.find(n => n.id === prevNodeId),
                        target: nodeData,
                        weight: weight,
                        weightGradient: weightGradient,
                    });
                }
            }
        }
    });

    // --- Draw Links ---
    const linkSelection = g.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(links)
        .enter().append("line")
        .attr("class", d => `link ${d.weight >= 0 ? 'positive-weight' : 'negative-weight'}`)
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y)
        .style("stroke-width", d => vizOptions.showWeights ? Math.min(5, Math.abs(d.weight * 2) + 0.5) : 1) // Scale width by weight if shown
        .style("display", vizOptions.showWeights ? null : "none") // Hide if weights checkbox unchecked
        .on("mouseover", (event, d) => {
            d3.select(event.currentTarget).classed("hovered", true).raise(); // Add class and bring to front
            tooltip.transition().duration(200).style("opacity", .9);
            let tooltipText = `Weight: ${formatValue(d.weight)}`;
            if (d.weightGradient !== null) {
                tooltipText += `\nGradient (dL/dW): ${formatValue(d.weightGradient)}`;
            }
            tooltip.html(tooltipText.replace(/\n/g, '<br/>'))
                   .style("left", (event.pageX + 10) + "px")
                   .style("top", (event.pageY - 15) + "px");
        })
        .on("mouseout", (event, d) => {
            d3.select(event.currentTarget).classed("hovered", false);
            tooltip.transition().duration(500).style("opacity", 0);
        });


    // --- Draw Nodes ---
    const nodeSelection = g.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(nodes)
        .enter().append("g")
        .attr("class", d => `node ${d.layerType} ${activationColorMapping[d.activation] || ''}`)
        .attr("transform", d => `translate(${d.x},${d.y})`);

    nodeSelection.append("circle")
        .attr("r", nodeRadius)
        .on("mouseover", (event, d) => {
            d3.select(event.currentTarget).classed("hovered", true);
            tooltip.transition().duration(200).style("opacity", .9);
            let tooltipText = `Layer: ${d.layerIndex}, Node: ${d.nodeIndex}`;
            if (d.activation) tooltipText += `\nActivation Fn: ${d.activation.toUpperCase()}`;
            if (d.logitValue !== null) tooltipText += `\nLogit (Z): ${formatValue(d.logitValue)}`;
            if (d.activationValue !== null) tooltipText += `\nActivation (A): ${formatValue(d.activationValue)}`;
            if (d.bias !== null) tooltipText += `\nBias: ${formatValue(d.bias)}`;
            if (d.deltaActivation !== null) tooltipText += `\nError Signal (dL/dA): ${formatValue(d.deltaActivation)}`;
             if (d.biasGradient !== null) tooltipText += `\nBias Gradient (dL/dB): ${formatValue(d.biasGradient)}`;

            tooltip.html(tooltipText.replace(/\n/g, '<br/>'))
                   .style("left", (event.pageX + 10) + "px")
                   .style("top", (event.pageY - 15) + "px");
        })
        .on("mouseout", (event, d) => {
            d3.select(event.currentTarget).classed("hovered", false);
            tooltip.transition().duration(500).style("opacity", 0);
        });

    // Add text inside nodes (Activation Value) if option is enabled
    nodeSelection.append("text")
        .attr("class", "node-value")
        .attr("dy", "0.3em") // Vertically center
        .attr("text-anchor", "middle")
        .text(d => formatValue(d.activationValue))
        .style("display", vizOptions.showValues ? null : "none"); // Hide if values checkbox unchecked

    // Add Layer Labels
    const layerLabels = layerConfig.map((layer, index) => ({
        x: index * layerGap,
        y: -margin.top / 2, // Position above the nodes
        text: getLayerType(index, numLayers).replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase()) + ` (${layer.nodes})`
    }));

    g.append("g")
        .attr("class", "layer-labels")
        .selectAll("text")
        .data(layerLabels)
        .enter().append("text")
        .attr("x", d => d.x)
        .attr("y", d => d.y)
        .attr("text-anchor", "middle")
        .style("font-weight", "bold")
        .text(d => d.text);

    // Apply current options state immediately after drawing
    applyVisualizationOptionsStyle();
}


// --- Options Handling ---

function setVisualizationOptions(options) {
    vizOptions = options;
    // Re-apply styles based on the new options
    applyVisualizationOptionsStyle();
}

function applyVisualizationOptionsStyle() {
     if (!svg) return; // Check if SVG exists

    // Toggle visibility based on options
    svg.selectAll(".link")
       .style("display", vizOptions.showWeights ? null : "none");
       // Optionally adjust stroke width again if needed based on weight value vs constant
       // .style("stroke-width", d => vizOptions.showWeights ? Math.min(5, Math.abs(d.weight * 2) + 0.5) : 1);

    svg.selectAll(".node-value") // Text inside nodes
       .style("display", vizOptions.showValues ? null : "none");

    // Could add similar logic for bias labels or gradient indicators if they were implemented
}


// --- Initial Call / Placeholder ---
// updateVisualization(null); // Show "Network not initialized" initially
// Or wait for the first call from script.js after initialization