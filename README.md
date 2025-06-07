Here is the implementation of linear regression from scratch

model : Find a set of parameters fit the data best on y = mx + b
Goal -> minimize the loss function with MSE

Process :

Initialization : We define random parameters for the model LinearRegression()
Inference Part : We expect x -> input, y -> output
Training Part : We try to improve the model by updating the parameters based on the loss function
Optimization : Gradient Descent -> w, b = w - a * dw, b - a * db

How to compute the gradient of the loss function with respect to the parameters ? ie. dL/dw, dL/db

First, we need to define the loss function
1. MSE (Mean Square Error) = 1/n * sum (y_pred - y)**2

Second, we need to compute the derivative
2a. dL_db = -2/n * sum(y[i]-y_pred[i]), dL/dW = -2
2b. dL_dw = -2/n * sum((x[i]*(y_pred[i] - y[i]))for i in range(size))

Third, we need to update the parameters
3. w, b = w - a * dw, b - a * db

Function : 
    def forward(self, x) :
    def loss(self, y_pred, y_true) :
    def gradient(self, x, y_pred, y) :
    def update(self, grad_w, grad_b) :

Training Loop : 
In each iteration, we need to :
1. Forward pass : Compute the prediction
2. Compute the loss
3. Backward pass : Compute the gradient
4. Update the parameters

