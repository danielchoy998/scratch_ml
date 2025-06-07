import random



random.seed(42)

class LinearRegression :
    def __init__(self,learning_rate : float = 0.001):
        self.weight = random.random()
        self.bias = random.random()
        self.learning_rate = learning_rate
    
    def update(self, grad_w, grad_b) :
        self.weight -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b
    
    def forward(self, x : list[float]) -> list[float]:
        y_pred = []
        for i in range(len(x)):
            y_pred.append(self.weight * x[i] + self.bias)
        return y_pred
    
    def loss(self, y_pred : list[float], y_true : list[float]) :
        size = len(y_pred)
        squared_error = []
        
        for i in range(size):
            error = y_pred[i]-y_true[i]
            squared_error.append(error**2)
        
        mse = 1/size * sum(squared_error)

        return mse

    def gradient(self, x, y_pred, y):
        size = len(x)
        dL_dw = - 2/size * sum((y_pred[i] - y[i]) * x[i] for i in range(size))
        dL_db = - 2/size * sum(y_pred[i] - y[i] for i in range(size))
        return dL_dw, dL_db

x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [2.3, 4.7, 6.9, 9.1, 11.3]

model = LinearRegression()
print(f"Initial weight : {model.weight}, Initial bias : {model.bias}")
y_pred = model.forward(x)

old_loss = model.loss(y_pred, y)
print(f"Initial loss : {old_loss}")

grad_w, grad_b = model.gradient(x,y_pred,y)
print(grad_w, grad_b)

model.update(grad_w, grad_b)
print(f"Updated weight : {model.weight}, Updated bias : {model.bias}")

# new prediction
y_pred = model.forward(x)
new_loss = model.loss(y_pred, y)
print(f"Updated loss : {new_loss}")
print(f"Loss decreased from {old_loss} to {new_loss}")

for i in range(1000):
    y_pred = model.forward(x)
    old_loss = model.loss(y_pred, y)
    grad_w, grad_b= model.gradient(x,y_pred,y)
    model.update(grad_w, grad_b)
    new_loss = model.loss(y_pred, y)
    print(f"Iteration {i+1} : Loss = {new_loss}")

