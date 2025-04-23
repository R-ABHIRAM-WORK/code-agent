import numpy as np
import math

# Define activation function names as constants
class ActivationFunctions:
    RELU = 'relu'
    GELU = 'gelu'
    SOFTMAX = 'softmax'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'
    LINEAR = 'linear'

    @staticmethod
    def get_all():
        """Returns a list of all available activation function names."""
        return [
            ActivationFunctions.RELU,
            ActivationFunctions.GELU,
            ActivationFunctions.SOFTMAX,
            ActivationFunctions.SIGMOID,
            ActivationFunctions.TANH,
            ActivationFunctions.LINEAR,
        ]

    @staticmethod
    def apply(name, z):
        """Applies the specified activation function."""
        z = np.array(z) # Ensure input is numpy array
        if name == ActivationFunctions.RELU:
            return np.maximum(0, z)
        elif name == ActivationFunctions.SIGMOID:
            # Clip to avoid overflow/underflow in exp
            z_clipped = np.clip(z, -500, 500)
            return 1 / (1 + np.exp(-z_clipped))
        elif name == ActivationFunctions.TANH:
            return np.tanh(z)
        elif name == ActivationFunctions.SOFTMAX:
            # Stable softmax: subtract max for numerical stability
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        elif name == ActivationFunctions.GELU:
            # Using the numpy/math approximation for GELU
            return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
        elif name == ActivationFunctions.LINEAR:
            return z
        else:
            raise ValueError(f"Unknown activation function: {name}")

    @staticmethod
    def derivative(name, a=None, z=None):
        """
        Computes the derivative of the activation function.
        Requires either the activation output 'a' or the pre-activation 'z'.
        'z' is needed for ReLU and GELU derivatives.
        """
        if name == ActivationFunctions.RELU:
            if z is None: raise ValueError("ReLU derivative requires 'z'")
            z = np.array(z)
            return np.where(z > 0, 1.0, 0.0)
        elif name == ActivationFunctions.SIGMOID:
            if a is None: raise ValueError("Sigmoid derivative requires 'a'")
            a = np.array(a)
            return a * (1 - a)
        elif name == ActivationFunctions.TANH:
            if a is None: raise ValueError("Tanh derivative requires 'a'")
            a = np.array(a)
            return 1 - a**2
        elif name == ActivationFunctions.SOFTMAX:
            # The derivative of softmax is typically combined with the loss function derivative.
            # For dL/dZ in backprop with Cross-Entropy loss, it simplifies to (A - Y).
            # Returning 1 as a placeholder; this function might not be directly called
            # in the standard backprop implementation for softmax output layer.
            return 1.0 # Placeholder
        elif name == ActivationFunctions.GELU:
            # Derivative of the GELU approximation
            if z is None: raise ValueError("GELU derivative requires 'z'")
            z = np.array(z)
            inner_const = np.sqrt(2 / np.pi)
            inner_term = inner_const * (z + 0.044715 * z**3)
            tanh_inner = np.tanh(inner_term)
            d_inner_dz = inner_const * (1 + 3 * 0.044715 * z**2)
            sech_sq = 1 - tanh_inner**2
            d_gelu_dz = 0.5 * (1 + tanh_inner) + 0.5 * z * sech_sq * d_inner_dz
            return d_gelu_dz
        elif name == ActivationFunctions.LINEAR:
             # Derivative is 1, needs shape information. Return 1.0 scalar or array?
             # Let's return scalar 1.0, backprop will handle broadcasting if needed.
             # Or return np.ones_like(a) if 'a' is provided.
             if a is not None:
                 return np.ones_like(np.array(a))
             elif z is not None:
                 return np.ones_like(np.array(z))
             else:
                 # This case should ideally not happen if called correctly
                 return 1.0 # Default scalar
        else:
            raise ValueError(f"Unknown activation function derivative: {name}")


class NeuralNetwork:
    def __init__(self, layer_config):
        """
        Initializes the neural network based on the layer configuration.

        Args:
            layer_config: A list of dictionaries, where each dictionary represents a layer
                          and contains 'nodes' (int) and 'activation' (str or None).
                          Example: [{'nodes': 784, 'activation': None},  # Input layer
                                    {'nodes': 128, 'activation': 'relu'}, # Hidden layer
                                    {'nodes': 10, 'activation': 'softmax'}] # Output layer
        """
        if not layer_config or len(layer_config) < 2:
            raise ValueError("Network must have at least an input and an output layer.")

        self.layer_config = layer_config
        self.num_layers = len(layer_config) # Total number of layers including input
        self.weights = [] # List to store weight matrices (NumPy arrays)
        self.biases = []  # List to store bias vectors (NumPy arrays)
        # Store activation function names for each layer (index corresponds to layer index)
        self.activations = [layer['activation'] for layer in layer_config]

        # Initialize weights and biases for layers 1 to L (L = num_layers - 1)
        # Layer 0 is input, no weights/biases leading *into* it in this context.
        for i in range(1, self.num_layers):
            nodes_in = self.layer_config[i-1]['nodes']
            nodes_out = self.layer_config[i]['nodes']
            activation_curr = self.activations[i] # Activation of the current layer (i)

            # Choose initialization based on the activation function of the *current* layer
            if activation_curr in [ActivationFunctions.RELU, ActivationFunctions.GELU]:
                # He initialization (good for ReLU-like activations)
                std_dev = np.sqrt(2.0 / nodes_in)
            else:
                # Xavier/Glorot initialization (good for sigmoid, tanh)
                # std_dev = np.sqrt(1.0 / nodes_in) # Simpler version
                std_dev = np.sqrt(2.0 / (nodes_in + nodes_out)) # More common version

            # Initialize weights with small random values ~ N(0, std_dev^2)
            # Shape: (nodes_out, nodes_in)
            w = np.random.randn(nodes_out, nodes_in) * std_dev
            # Initialize biases to zeros
            # Shape: (nodes_out, 1) to facilitate broadcasting
            b = np.zeros((nodes_out, 1))

            self.weights.append(w)
            self.biases.append(b)

        print(f"NN Initialized. Layers: {self.num_layers}")
        print(f"Layer Config: {self.layer_config}")
        print(f"Weight shapes: {[w.shape for w in self.weights]}")
        print(f"Bias shapes: {[b.shape for b in self.biases]}")


    def forward_pass(self, input_data):
        """
        Performs a single forward pass through the network.

        Args:
            input_data: A list or NumPy array representing the input features for one sample.

        Returns:
            A dictionary containing:
            - 'inputs': The input data (as list).
            - 'layer_outputs': List of activation outputs for each layer (including input), as lists.
            - 'logits': List of pre-activation outputs (Z) for each layer (excluding input), as lists.
            - 'final_output': The final output of the network (as list).
            - '_internal_cache': Dictionary with NumPy arrays {'A': [A0..AL], 'Z': [Z1..ZL]} for backprop.
        """
        if not isinstance(input_data, np.ndarray):
             input_data = np.array(input_data)

        # Ensure input is a column vector (features, 1)
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)

        if input_data.shape[0] != self.layer_config[0]['nodes']:
             raise ValueError(f"Input data features ({input_data.shape[0]}) doesn't match input layer nodes ({self.layer_config[0]['nodes']})")
        if input_data.shape[1] != 1:
             raise ValueError("This implementation supports single sample forward pass (input shape should be (nodes, 1))")

        # Cache for intermediate results (needed for backpropagation)
        cache_A = [input_data] # List of activations, A[0] = input
        cache_Z = []           # List of pre-activation values (logits)

        A_curr = input_data
        # Iterate through layers from 1 to L (index 0 to L-1 for weights/biases)
        for i in range(self.num_layers - 1):
            W = self.weights[i] # Shape: (nodes_out, nodes_in)
            b = self.biases[i]  # Shape: (nodes_out, 1)
            activation_name = self.activations[i+1] # Activation for layer i+1

            # --- Linear Step ---
            # Z = W @ A_prev + b
            # A_curr shape: (nodes_in, 1)
            # Z shape: (nodes_out, 1)
            Z = np.dot(W, A_curr) + b
            cache_Z.append(Z)

            # --- Activation Step ---
            # A = g(Z)
            A_curr = ActivationFunctions.apply(activation_name, Z)
            cache_A.append(A_curr)

        final_output = A_curr

        # Prepare results for API (convert numpy arrays to lists)
        return {
            "inputs": input_data.flatten().tolist(),
            "layer_outputs": [a.flatten().tolist() for a in cache_A],
            "logits": [z.flatten().tolist() for z in cache_Z],
            "final_output": final_output.flatten().tolist(),
            # Keep internal cache with NumPy arrays for efficiency in backprop
            "_internal_cache": {'A': cache_A, 'Z': cache_Z}
        }


    def backward_pass(self, forward_pass_results, target_data, learning_rate=0.01):
        """
        Performs backward propagation and updates network parameters.

        Args:
            forward_pass_results: The dictionary returned by `forward_pass`.
            target_data: List or NumPy array of target values for the single sample.
            learning_rate: The learning rate for parameter updates.

        Returns:
            A dictionary containing:
            - 'gradients_w': List of weight gradients (dL/dW) for each layer, as lists.
            - 'gradients_b': List of bias gradients (dL/db) for each layer, as lists.
            - 'delta_activations': List of error signals w.r.t activations (dL/dA) per layer, as lists.
            - 'updated_weights': List of weights after update, as lists.
            - 'updated_biases': List of biases after update, as lists.
        """
        if not isinstance(target_data, np.ndarray):
            target_data = np.array(target_data)

        # Ensure target is a column vector (output_nodes, 1)
        if target_data.ndim == 1:
            target_data = target_data.reshape(-1, 1)

        if target_data.shape[0] != self.layer_config[-1]['nodes']:
             raise ValueError(f"Target data size ({target_data.shape[0]}) doesn't match output layer nodes ({self.layer_config[-1]['nodes']})")
        if target_data.shape[1] != 1:
             raise ValueError("This implementation supports single sample backward pass (target shape should be (nodes, 1))")

        # Retrieve cached NumPy arrays from forward pass
        if "_internal_cache" not in forward_pass_results:
             raise ValueError("Internal cache from forward pass is missing.")
        cache_A = forward_pass_results["_internal_cache"]['A'] # [A0, A1, ..., AL]
        cache_Z = forward_pass_results["_internal_cache"]['Z'] # [Z1, ..., ZL]
        # L = self.num_layers - 1 (index of the last layer)

        # Initialize gradients dictionary (will store NumPy arrays)
        gradients = {'dW': [None] * len(self.weights), 'db': [None] * len(self.biases)}
        # Store dL/dA for visualization/debugging (NumPy arrays)
        delta_activations = [None] * self.num_layers

        # --- Step 1: Compute initial delta (dL/dZ) at the output layer (L) ---
        A_L = cache_A[-1] # Output activation (shape: nodes_L, 1)
        Z_L = cache_Z[-1] # Output logits (shape: nodes_L, 1)
        Y = target_data   # True labels (shape: nodes_L, 1)
        output_activation = self.activations[-1] # Activation name of the output layer

        # Calculate dL/dZ_L based on loss function and output activation
        if output_activation == ActivationFunctions.SOFTMAX:
            # Assumes Cross-Entropy Loss is used with Softmax
            # dL/dZ_L = A_L - Y
            dZ_L = A_L - Y
            # dL/dA_L = -Y / np.clip(A_L, 1e-9, 1.0) # If needed explicitly
            # delta_activations[-1] = dL_dA_L
        else:
            # Assumes Mean Squared Error (MSE) Loss: L = 0.5 * sum((A_L - Y)^2)
            # dL/dA_L = A_L - Y
            dL_dA_L = A_L - Y
            delta_activations[-1] = dL_dA_L # Store dL/dA[L]
            # dA_L/dZ_L = g'(Z_L)
            # Need Z_L for ReLU/GELU derivative
            dA_L_dZ_L = ActivationFunctions.derivative(output_activation, a=A_L, z=Z_L)
            # dL/dZ_L = dL/dA_L * dA_L/dZ_L
            dZ_L = dL_dA_L * dA_L_dZ_L

        # --- Step 2: Calculate gradients for the output layer (L) ---
        # Layer L corresponds to index L-1 in weights/biases lists
        # Activation from previous layer (L-1) is A_{L-1} at index L-1 in cache_A
        A_prev = cache_A[-2] # Shape: (nodes_{L-1}, 1)
        m = 1 # Batch size is 1 for this interactive simulator

        # dW_L = (1/m) * dZ_L @ A_{L-1}.T
        dW_L = (1/m) * np.dot(dZ_L, A_prev.T) # Shape: (nodes_L, nodes_{L-1})
        # db_L = (1/m) * sum(dZ_L along batch axis)
        db_L = (1/m) * np.sum(dZ_L, axis=1, keepdims=True) # Shape: (nodes_L, 1)

        gradients['dW'][-1] = dW_L
        gradients['db'][-1] = db_L

        # --- Step 3: Propagate delta backwards through hidden layers ---
        dZ_curr = dZ_L # Start with the delta from the output layer

        # Loop from layer L-1 down to layer 1
        # Python range goes from num_layers-2 down to 0 (inclusive)
        # This corresponds to indices of weights/biases matrices
        for l in range(self.num_layers - 2, -1, -1):
            # l = index for W[l], b[l] (connecting layer l to l+1)
            # Layer l+1 activation: cache_A[l+1], logits: cache_Z[l]
            # Layer l activation: cache_A[l]
            # Weight matrix *after* this layer: W_{l+1} (index l+1 in self.weights)

            activation_name = self.activations[l+1] # Activation function of layer l+1
            A_curr = cache_A[l+1] # Activation of layer l+1 (shape: nodes_{l+1}, 1)
            Z_curr = cache_Z[l]   # Logits of layer l+1 (shape: nodes_{l+1}, 1)
            A_prev = cache_A[l]   # Activation from layer l (shape: nodes_l, 1)
            W_next = self.weights[l+1] # Weights connecting l+1 to l+2 (shape: nodes_{l+2}, nodes_{l+1})

            # Calculate dL/dA_curr (delta for activation of current layer l+1)
            # dA_curr = W_{next}.T @ dZ_{next}
            dA_curr = np.dot(W_next.T, dZ_curr) # Shape: (nodes_{l+1}, 1)
            delta_activations[l+1] = dA_curr # Store dL/dA[l+1]

            # Calculate dL/dZ_curr (delta for logits of current layer l+1)
            # dZ_curr = dA_curr * g'(Z_curr)
            # Need Z_curr for ReLU/GELU derivative
            dZ_curr = dA_curr * ActivationFunctions.derivative(activation_name, a=A_curr, z=Z_curr) # Shape: (nodes_{l+1}, 1)

            # Calculate gradients for weights W[l] and biases b[l]
            # dW[l] = (1/m) * dZ_curr @ A_prev.T
            dW = (1/m) * np.dot(dZ_curr, A_prev.T) # Shape: (nodes_{l+1}, nodes_l)
            # db[l] = (1/m) * sum(dZ_curr along batch axis)
            db = (1/m) * np.sum(dZ_curr, axis=1, keepdims=True) # Shape: (nodes_{l+1}, 1)

            gradients['dW'][l] = dW
            gradients['db'][l] = db

            # dZ_curr becomes the dZ for the *next* iteration (previous layer)

        # --- Step 4: Update weights and biases ---
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients['dW'][i]
            self.biases[i] -= learning_rate * gradients['db'][i]

        # --- Prepare results for API (convert NumPy arrays to lists) ---
        gradients_w_list = [dw.tolist() for dw in gradients['dW']]
        # Flatten biases gradients for simpler JSON structure
        gradients_b_list = [db.flatten().tolist() for db in gradients['db']]
        # Flatten delta_activations, handle None for input layer
        delta_activations_list = [da.flatten().tolist() if da is not None else None
                                  for da in delta_activations]
        updated_weights_list = [w.tolist() for w in self.weights]
        # Flatten updated biases
        updated_biases_list = [b.flatten().tolist() for b in self.biases]

        return {
            "gradients_w": gradients_w_list,
            "gradients_b": gradients_b_list,
            "delta_activations": delta_activations_list,
            "updated_weights": updated_weights_list,
            "updated_biases": updated_biases_list
        }


    def get_structure(self):
        """
        Returns a serializable dictionary representing the network's structure.
        Weights and biases are converted from NumPy arrays to nested lists.
        """
        weights_list = [w.tolist() for w in self.weights]
        # Flatten biases for simpler JSON structure (e.g., [0.1, 0.2] instead of [[0.1], [0.2]])
        biases_list = [b.flatten().tolist() for b in self.biases]

        return {
            'layer_config': self.layer_config, # Already serializable
            'weights': weights_list,
            'biases': biases_list,
            'activations': self.activations # List of strings, already serializable
        }

# Example Usage (for standalone testing)
if __name__ == '__main__':
    print("--- Running Neural Network Standalone Test ---")

    # Config: Input (2 nodes) -> Hidden (3 nodes, ReLU) -> Output (1 node, Sigmoid)
    config_sigmoid = [
        {'nodes': 2, 'activation': None},
        {'nodes': 3, 'activation': ActivationFunctions.RELU},
        {'nodes': 1, 'activation': ActivationFunctions.SIGMOID}
    ]
    print("\nInitializing Sigmoid Output Network...")
    nn_sigmoid = NeuralNetwork(config_sigmoid)
    # print("\nInitial Structure:")
    # print(nn_sigmoid.get_structure())

    input_data_sig = np.array([0.8, -0.2])
    target_data_sig = np.array([0.95]) # Target close to 1
    print(f"\nInput Data: {input_data_sig.tolist()}")
    print(f"Target Data: {target_data_sig.tolist()}")

    print("\nPerforming Forward Pass...")
    forward_results_sig = nn_sigmoid.forward_pass(input_data_sig)
    print(f"  Final Output: {forward_results_sig['final_output']}")

    print("\nPerforming Backward Pass (learning_rate=0.1)...")
    backward_results_sig = nn_sigmoid.backward_pass(forward_results_sig, target_data_sig, learning_rate=0.1)
    # print(f"  Weight Gradients (dW): {backward_results_sig['gradients_w']}") # Can be verbose
    # print(f"  Bias Gradients (db): {backward_results_sig['gradients_b']}")
    print("  Backward pass completed.")

    print("\nPerforming Forward Pass After Update...")
    forward_results_sig_updated = nn_sigmoid.forward_pass(input_data_sig)
    print(f"  New Final Output: {forward_results_sig_updated['final_output']}")
    print("--- Sigmoid Test Complete ---")


    # Config: Input (4) -> Hidden (5, GELU) -> Output (3, Softmax)
    config_softmax = [
        {'nodes': 4, 'activation': None},
        {'nodes': 5, 'activation': ActivationFunctions.GELU},
        {'nodes': 3, 'activation': ActivationFunctions.SOFTMAX} # 3 output classes
    ]
    print("\n\nInitializing Softmax Output Network...")
    nn_softmax = NeuralNetwork(config_softmax)

    input_data_soft = np.random.rand(4) * 2 - 1 # Random input between -1 and 1
    # Target is one-hot encoded for class 1 (index 1)
    target_data_soft = np.array([0.0, 1.0, 0.0])
    print(f"\nInput Data: {input_data_soft.tolist()}")
    print(f"Target Data (One-Hot): {target_data_soft.tolist()}")

    print("\nPerforming Forward Pass...")
    forward_softmax = nn_softmax.forward_pass(input_data_soft)
    print(f"  Final Output (Probabilities): {forward_softmax['final_output']}")

    print("\nPerforming Backward Pass (learning_rate=0.05)...")
    backward_softmax = nn_softmax.backward_pass(forward_softmax, target_data_soft, learning_rate=0.05)
    print("  Backward pass completed.")

    print("\nPerforming Forward Pass After Update...")
    forward_softmax_updated = nn_softmax.forward_pass(input_data_soft)
    print(f"  New Final Output: {forward_softmax_updated['final_output']}")
    print("--- Softmax Test Complete ---")