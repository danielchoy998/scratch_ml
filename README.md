# Linear Regression from Scratch

## Overview
Implementation of linear regression from scratch using gradient descent optimization.

**Model**: Find parameters that best fit the data using the equation `y = mx + b`  
**Goal**: Minimize the loss function using Mean Squared Error (MSE)

## Mathematical Foundation

### Loss Function
**Mean Squared Error (MSE)**:
```
L = (1/n) * Σ(y_pred - y_true)²
```

### Gradient Computation
To minimize the loss, we compute partial derivatives:

**Gradient with respect to bias (b)**:
```
∂L/∂b = (-2/n) * Σ(y_true - y_pred)
```

**Gradient with respect to weight (w)**:
```
∂L/∂w = (-2/n) * Σ(x * (y_true - y_pred))
```

### Parameter Update
Using gradient descent with learning rate `α`:
```
w = w - α * ∂L/∂w
b = b - α * ∂L/∂b
```

## Implementation Structure

### Core Functions
```python
def forward(self, x):
    # Forward pass: compute predictions

def loss(self, y_pred, y_true):
    # Compute MSE loss

def gradient(self, x, y_pred, y_true):
    # Compute gradients for w and b

def update(self, grad_w, grad_b):
    # Update parameters using gradients
```

### Training Process
1. **Initialization**: Define random parameters for LinearRegression()
2. **Inference**: Input `x` → Output `y`
3. **Training**: Update parameters based on loss function
4. **Optimization**: Apply gradient descent

### Training Loop
For each iteration:
1. **Forward pass**: Compute predictions
2. **Loss computation**: Calculate MSE
3. **Backward pass**: Compute gradients
4. **Parameter update**: Apply gradient descent

