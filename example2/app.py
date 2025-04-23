from flask import Flask, render_template, request, jsonify
import json
from neural_network import NeuralNetwork, ActivationFunctions

app = Flask(__name__)

# Placeholder for the neural network instance
network = None

# Placeholder activation functions (replace with actual implementations)
class ActivationFunctions:
    RELU = 'relu'
    GELU = 'gelu'
    SOFTMAX = 'softmax'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    LINEAR = 'linear' # For output layer or specific cases

    @staticmethod
    def get_all():
        return [
            ActivationFunctions.RELU,
            ActivationFunctions.GELU,
            ActivationFunctions.SOFTMAX,
            ActivationFunctions.SIGMOID,
            ActivationFunctions.TANH,
            ActivationFunctions.LINEAR,
        ]

# Placeholder Neural Network class (replace with actual implementation in neural_network.py)
class NeuralNetwork:
    def __init__(self, layer_config):
        self.layer_config = layer_config
        self.weights = []
        self.biases = []
        self.activations = [] # Store activation function names per layer
        self._initialize_network()
        print(f"NN Initialized with config: {layer_config}")

    def _initialize_network(self):
        # Dummy initialization based on config
        # layer_config = [{'nodes': count, 'activation': name}, ...]
        self.weights = []
        self.biases = []
        self.activations = []
        if not self.layer_config:
            return

        # Input layer doesn't have weights/biases coming *into* it in this context
        # It only defines the number of features for the first weight matrix.
        self.activations.append(None) # No activation for input layer itself

        for i in range(len(self.layer_config) - 1):
            nodes_in = self.layer_config[i]['nodes']
            nodes_out = self.layer_config[i+1]['nodes']
            # Replace with actual random initialization (e.g., Xavier/He)
            self.weights.append([[0.1 for _ in range(nodes_in)] for _ in range(nodes_out)])
            self.biases.append([0.1 for _ in range(nodes_out)])
            self.activations.append(self.layer_config[i+1]['activation'])

        print(f"Weights shapes: {[ (len(w), len(w[0])) for w in self.weights]}")
        print(f"Biases shapes: {[len(b) for b in self.biases]}")
        print(f"Activations: {self.activations}")


    def forward_pass(self, input_data):
        # Dummy forward pass
        print(f"Forward pass with input: {input_data}")
        # Validate input size
        if len(input_data) != self.layer_config[0]['nodes']:
             raise ValueError(f"Input data size ({len(input_data)}) doesn't match input layer nodes ({self.layer_config[0]['nodes']})")

        layer_outputs = [input_data] # List to store output of each layer
        logits = [] # Store pre-activation outputs (logits)

        current_output = input_data
        for i in range(len(self.weights)):
            # Dummy matrix multiplication and addition
            z = [sum(w_ij * x_j for w_ij, x_j in zip(self.weights[i][neuron_idx], current_output)) + self.biases[i][neuron_idx]
                 for neuron_idx in range(len(self.weights[i]))]
            logits.append(z)

            # Dummy activation
            activation_func = self.activations[i+1] # i+1 because activations[0] is None (input)
            if activation_func == ActivationFunctions.RELU:
                current_output = [max(0, val) for val in z]
            elif activation_func == ActivationFunctions.SOFTMAX:
                 # Basic softmax, numerically unstable, replace with stable version
                 exp_z = [2.71828**val for val in z]
                 sum_exp_z = sum(exp_z)
                 current_output = [val / sum_exp_z for val in exp_z]
            # Add other activation functions here (GELU, SIGMOID, TANH, LINEAR)
            elif activation_func == ActivationFunctions.LINEAR:
                 current_output = z # No activation
            else: # Default to linear for simplicity in placeholder
                 current_output = z

            layer_outputs.append(current_output)

        print(f"Forward pass result: {current_output}")
        return {
            "inputs": input_data,
            "layer_outputs": layer_outputs, # Includes input layer output
            "logits": logits, # Logits for each layer (except input)
            "final_output": current_output
        }

    def backward_pass(self, forward_pass_results, target_data, learning_rate=0.01):
        # Dummy backward pass
        print(f"Backward pass with targets: {target_data}, learning rate: {learning_rate}")
        # Requires forward pass results (activations, logits)
        # Calculate gradients (dummy values for now)
        # Update weights and biases (dummy updates)

        gradients_w = [[[0.01 for _ in w_row] for w_row in w] for w in self.weights] # dLoss/dW
        gradients_b = [[0.01 for _ in b] for b in self.biases] # dLoss/dB
        delta_activations = [[0.01 for _ in layer] for layer in forward_pass_results["layer_outputs"]] # dLoss/dA

        # Dummy weight/bias update step
        new_weights = []
        for layer_w, layer_grad_w in zip(self.weights, gradients_w):
            new_layer_w = []
            for r_idx, row in enumerate(layer_w):
                 new_row = [w - learning_rate * grad for w, grad in zip(row, layer_grad_w[r_idx])]
                 new_layer_w.append(new_row)
            new_weights.append(new_layer_w)
        self.weights = new_weights

        new_biases = []
        for layer_b, layer_grad_b in zip(self.biases, gradients_b):
             new_layer_b = [b - learning_rate * grad for b, grad in zip(layer_b, layer_grad_b)]
             new_biases.append(new_layer_b)
        self.biases = new_biases


        print("Weights and biases updated (dummy update)")
        return {
            "gradients_w": gradients_w,
            "gradients_b": gradients_b,
            "delta_activations": delta_activations, # Error signal propagated back
            "updated_weights": self.weights,
            "updated_biases": self.biases
        }

    def get_structure(self):
        # Return a serializable representation of the network structure
        return {
            'layer_config': self.layer_config,
            'weights': self.weights,
            'biases': self.biases,
            'activations': self.activations # List of activation function names per layer
        }

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html', activation_functions=ActivationFunctions.get_all())

@app.route('/api/initialize', methods=['POST'])
def initialize_network():
    """Initializes the neural network based on user configuration."""
    global network
    try:
        config = request.json
        print(f"Received config for initialization: {config}")

        # Basic validation
        if not config or 'layers' not in config or not isinstance(config['layers'], list) or len(config['layers']) < 2:
             return jsonify({"error": "Invalid network configuration. Need at least input and output layers."}), 400

        layer_config = []
        for i, layer in enumerate(config['layers']):
            if not isinstance(layer, dict) or 'nodes' not in layer or 'activation' not in layer:
                 return jsonify({"error": f"Invalid format for layer {i}."}), 400
            try:
                nodes = int(layer['nodes'])
                if nodes <= 0:
                    raise ValueError("Node count must be positive.")
            except ValueError as e:
                 return jsonify({"error": f"Invalid node count for layer {i}: {e}"}), 400

            activation = layer['activation']
            if activation not in ActivationFunctions.get_all():
                 # Allow None only for input layer conceptually, but enforce selection for hidden/output
                 # The NN class handles internal logic of activation application
                 if i > 0 and activation is None:
                     return jsonify({"error": f"Activation function must be selected for layer {i}."}), 400
                 # Accept None for input layer if passed, though it's not used for computation there
                 # Or default input layer activation if needed by NN class structure

            layer_config.append({'nodes': nodes, 'activation': activation})


        # Create the Neural Network instance (using placeholder class)
        network = NeuralNetwork(layer_config)

        return jsonify(network.get_structure())

    except Exception as e:
        print(f"Error during initialization: {e}")
        # Log the full error traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.route('/api/forward_pass', methods=['POST'])
def forward_pass_api():
    """Performs a forward pass with the given input data."""
    global network
    if not network:
        return jsonify({"error": "Network not initialized."}), 400

    try:
        data = request.json
        print(f"Received data for forward pass: {data}")
        if 'input_data' not in data or not isinstance(data['input_data'], list):
            return jsonify({"error": "Missing or invalid 'input_data'."}), 400

        # Convert input data to floats
        try:
            input_data = [float(x) for x in data['input_data']]
        except ValueError:
             return jsonify({"error": "Input data must contain only numbers."}), 400


        # Perform forward pass using the network instance
        results = network.forward_pass(input_data)
        return jsonify(results)

    except ValueError as e: # Catch specific errors like input size mismatch
        print(f"Value error during forward pass: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500


@app.route('/api/backward_pass', methods=['POST'])
def backward_pass_api():
    """Performs a backward pass with the given target data."""
    global network
    if not network:
        return jsonify({"error": "Network not initialized."}), 400

    try:
        data = request.json
        print(f"Received data for backward pass: {data}")
        if 'target_data' not in data or not isinstance(data['target_data'], list):
            return jsonify({"error": "Missing or invalid 'target_data'."}), 400
        if 'forward_pass_results' not in data or not isinstance(data['forward_pass_results'], dict):
             return jsonify({"error": "Missing or invalid 'forward_pass_results'."}), 400

        learning_rate = float(data.get('learning_rate', 0.01)) # Default learning rate

        # Convert target data to floats
        try:
            target_data = [float(y) for y in data['target_data']]
        except ValueError:
             return jsonify({"error": "Target data must contain only numbers."}), 400

        # Validate target data size against output layer size
        output_layer_nodes = network.layer_config[-1]['nodes']
        if len(target_data) != output_layer_nodes:
            return jsonify({"error": f"Target data size ({len(target_data)}) doesn't match output layer nodes ({output_layer_nodes})."}), 400


        # Perform backward pass using the network instance
        # Pass the necessary results from the forward pass
        backward_results = network.backward_pass(
            data['forward_pass_results'],
            target_data,
            learning_rate
        )
        return jsonify(backward_results)

    except ValueError as e: # Catch specific errors
        print(f"Value error during backward pass: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Error during backward pass: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {e}"}), 500

if __name__ == '__main__':
    # Use a different port if 5000 is taken, enable debug for development
    app.run(debug=True, port=5001)