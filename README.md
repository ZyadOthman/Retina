# Retina - A Simple PyTorch Recreation in C++ with Intel GPU Support

## Overview

**Retina** is a lightweight recreation of key PyTorch functionalities using C++ with support for Intel GPUs via SYCL. The goal of this project is to provide an understanding of the fundamental building blocks of PyTorch while leveraging Intel's GPU acceleration. Retina includes custom implementations of essential PyTorch components like `Tensor`, neural network layers, and optimization routines.

This project demonstrates the creation of a simple feedforward neural network using Retina's API. The network is trained to approximate a nonlinear function using backpropagation and gradient descent.

## Features
- **C++ Implementation**: Core PyTorch components like `Tensor` and neural network modules are implemented from scratch in C++.
- **Intel GPU Acceleration**: Leverages SYCL for GPU acceleration on Intel hardware.
- **Neural Network Layers**: Includes fully connected (`Linear`) layers and activation functions like `Sigmoid`.
- **Optimization and Loss**: Supports Mean Squared Error (MSE) loss and Stochastic Gradient Descent (SGD) for optimization.
- **Customizable Neural Networks**: Easily define and train custom models using Retina’s API.

## Example Usage

### Defining a Model

```python
from nn.module import Module
from nn.modules.linear import Linear
from nn.activation import Sigmoid
from nn.loss import MSELoss
import tensor.Tensor as Tensor
from optim.optimizer import SGD
import random
import math

random.seed(1)

class MyModel(Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = Linear(1, 10)
        self.sigmoid = Sigmoid()
        self.fc2 = Linear(10, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out
```

### Training the Model

We train the model using a simple dataset of `x` values and their corresponding target `y` values (computed as the square of the sine of `x`).

```python
device = "sycl"  # Use SYCL for Intel GPU acceleration
epochs = 10

model = MyModel().to(device)
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=5)
loss_list = []
outputs_list = []

x_values = [...]
y_true = [math.pow(math.sin(x), 2) for x in x_values]

for epoch in range(epochs):
    for x, target in zip(x_values, y_true):
        x = Tensor([[x]]).T.to(device)
        target = Tensor([[target]]).T.to(device)

        outputs = model(x)

        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == epochs - 1:
            outputs_list.append(outputs.to("cpu")[0])

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss[0]:.4f}')
    loss_list.append(loss)

print(outputs_list)
print(y_true)
```

### Expected Output

After training for 10 epochs, the model approximates the target function and outputs the predicted values compared to the true `y` values.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/retina.git
   ```

2. Ensure you have the necessary Intel GPU drivers and SYCL environment set up. Follow Intel's guidelines for installing SYCL support.

3. Build the project using CMake (or your preferred build system) and link against Intel's DPC++ compiler for SYCL support.

## Contributing

Feel free to open issues or submit pull requests if you find any bugs or have suggestions for improvements.

---

Retina is a project aimed at demystifying the inner workings of PyTorch by recreating its core features and providing GPU acceleration using SYCL. This project is ideal for anyone looking to deepen their understanding of deep learning frameworks and explore GPU-accelerated programming.