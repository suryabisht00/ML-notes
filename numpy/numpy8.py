import numpy as np

"""
Neural Network Operations Explained
--------------------------------

1. Activation Functions:
   - Sigmoid: σ(x) = 1/(1 + e^(-x))
     * Range: (0,1)
     * Use: Binary classification output
   
   - ReLU: f(x) = max(0,x)
     * Range: [0,∞)
     * Use: Hidden layers, prevents vanishing gradient
"""
# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

"""
2. Layer Operations:
   - Forward Pass: Z = X·W + b
   - Activation: A = f(Z)
   - Shape Rules: 
     * Input (n_samples, n_features)
     * Weights (n_features, n_neurons)
     * Output (n_samples, n_neurons)
"""
# Forward pass simulation
X = np.random.randn(5, 3)  # 5 samples, 3 features
W1 = np.random.randn(3, 4)  # First layer weights
W2 = np.random.randn(4, 2)  # Second layer weights

# Layer computations
layer1 = relu(np.dot(X, W1))
layer2 = sigmoid(np.dot(layer1, W2))

print("\nInput shape:", X.shape)
print("Layer 1 output shape:", layer1.shape)
print("Layer 2 output shape:", layer2.shape)

"""
3. Batch Normalization:
   - Purpose: Stabilize learning process
   - Formula: y = (x - μ_batch)/sqrt(σ²_batch + ε)
   - Benefits: 
     * Reduces internal covariate shift
     * Allows higher learning rates
"""
# Batch normalization
batch_mean = np.mean(layer1, axis=0)
batch_var = np.var(layer1, axis=0)
batch_normalized = (layer1 - batch_mean) / np.sqrt(batch_var + 1e-8)
print("\nBatch normalized stats:")
print("Mean:", np.mean(batch_normalized, axis=0))
print("Var:", np.var(batch_normalized, axis=0))

# Add backpropagation example
def backward_pass(dY, cache):
    """
    Backward propagation implementation
    dY: Gradient of loss with respect to output
    cache: Stored values from forward pass
    """
    X, W, Z = cache
    dZ = dY * sigmoid_derivative(Z)
    dW = np.dot(X.T, dZ)
    dX = np.dot(dZ, W.T)
    return dX, dW
