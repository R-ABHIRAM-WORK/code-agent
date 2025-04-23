/**
 * simulation.js
 *
 * Contains functions for animating the forward and backward passes
 * on the network visualization SVG.
 * Relies on elements and classes set up by network-visualization.js.
 */

const animationSpeed = 500; // Milliseconds per step (layer or connection)

/**
 * Animates the flow of activation through the network during the forward pass.
 *
 * @param {Array} layerOutputs - Array of activation outputs for each layer (including input).
 * @param {Array} layerLogits - Array of pre-activation outputs (logits) for hidden/output layers.
 * @param {Object} networkStructure - The current network structure (needed for layer/node counts).
 */
async function animateForwardPass(layerOutputs, layerLogits, networkStructure) {
    if (!networkStructure || !layerOutputs) return;

    const numLayers = networkStructure.layer_config.length;
    const nodeBaseSelector = '.node';
    const linkBaseSelector = '.link';

    // Reset any previous animations
    d3.selectAll(`${nodeBaseSelector} circle`).classed('activating', false).style('opacity', 1);
    d3.selectAll(linkBaseSelector).classed('transmitting', false).style('opacity', 0.6); // Reset opacity

    // Function to animate a single layer's activation propagation
    const animateLayer = async (layerIndex) => {
        // Highlight nodes in the current layer being activated
        const currentLayerNodes = d3.selectAll(`${nodeBaseSelector}.layer-${layerIndex} circle`);
        currentLayerNodes.classed('activating', true);

        // Highlight links transmitting to the current layer
        if (layerIndex > 0) {
            const incomingLinks = d3.selectAll(linkBaseSelector)
                .filter(d => d.target.layerIndex === layerIndex); // Filter links ending at this layer

            incomingLinks.classed('transmitting', true)
                         .style('opacity', 1.0); // Make active links more visible

            // Wait for link animation
            await new Promise(resolve => setTimeout(resolve, animationSpeed / 2));

            // Reset link animation after a delay
             incomingLinks.classed('transmitting', false)
                          .transition().duration(animationSpeed / 2)
                          .style('opacity', 0.6); // Fade back
        }

         // Wait for node activation animation
        await new Promise(resolve => setTimeout(resolve, animationSpeed));

        // Reset node animation
        currentLayerNodes.classed('activating', false);

    };

    // Animate layer by layer
    for (let i = 0; i < numLayers; i++) {
        await animateLayer(i);
    }

    // Ensure final state is clean
    d3.selectAll(`${nodeBaseSelector} circle`).classed('activating', false);
    d3.selectAll(linkBaseSelector).classed('transmitting', false).style('opacity', 0.6); // Ensure all links are reset

    console.log("Forward pass animation complete.");
}


/**
 * Animates the backpropagation of error signals and gradient calculations.
 * (This is a simplified conceptual animation)
 *
 * @param {Object} backwardResults - Results from the backend backward pass API call.
 * @param {Object} networkStructure - The current network structure.
 */
async function animateBackwardPass(backwardResults, networkStructure) {
     if (!networkStructure || !backwardResults) return;

    const numLayers = networkStructure.layer_config.length;
    const nodeBaseSelector = '.node';
    const linkBaseSelector = '.link';

    // Reset any previous animations
    d3.selectAll(`${nodeBaseSelector} circle`).classed('back-activating', false).style('opacity', 1); // Use a different class?
    d3.selectAll(linkBaseSelector).classed('back-transmitting', false).style('opacity', 0.6);

    // Function to animate error signal propagation for a layer
    const animateLayerBackward = async (layerIndex) => {
        // Highlight nodes where error signal (dL/dA or dL/dZ) is calculated
        const currentLayerNodes = d3.selectAll(`${nodeBaseSelector}.layer-${layerIndex} circle`);
        currentLayerNodes.classed('back-activating', true) // Use a distinct class for backprop animation
                         .style('stroke', 'red'); // Example: Highlight border red

        // Highlight links involved in propagating error *from* this layer backwards
        if (layerIndex > 0) {
            const outgoingLinks = d3.selectAll(linkBaseSelector)
                .filter(d => d.target.layerIndex === layerIndex); // Links ending at this layer were used

             outgoingLinks.classed('back-transmitting', true)
                          .style('stroke', 'orange') // Example: Highlight links orange
                          .style('opacity', 1.0);

            // Wait for link animation
            await new Promise(resolve => setTimeout(resolve, animationSpeed / 2));

            // Reset link animation
             outgoingLinks.classed('back-transmitting', false)
                          .transition().duration(animationSpeed / 2)
                          .style('stroke', null) // Reset stroke color
                          .style('opacity', 0.6);
        }

        // Wait for node animation
        await new Promise(resolve => setTimeout(resolve, animationSpeed));

        // Reset node animation
        currentLayerNodes.classed('back-activating', false)
                         .style('stroke', null); // Reset border color
    };

    // Animate backwards from output layer to input layer
    for (let i = numLayers - 1; i >= 0; i--) {
        await animateLayerBackward(i);
    }

     // Ensure final state is clean
    d3.selectAll(`${nodeBaseSelector} circle`).classed('back-activating', false).style('stroke', null);
    d3.selectAll(linkBaseSelector).classed('back-transmitting', false).style('stroke', null).style('opacity', 0.6);

    console.log("Backward pass animation complete.");
}

// Note: These animation functions are conceptual. They highlight layers and potentially
// links involved in the respective passes. More detailed animations (e.g., showing
// individual value flows or gradient calculations) would require significantly
// more complex logic and potentially different data structures from the backend.
// Integration: These functions would typically be called from script.js after
// receiving the results from the API calls for forward/backward passes, potentially
// before the final call to updateVisualization which shows the static end result.