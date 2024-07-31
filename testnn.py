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

device = "sycl"
epochs = 10

model = MyModel().to(device)
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=5)
loss_list = []
outputs_list = []

x_values = [0. ,  0.4,  0.8,  1.2,  1.6,  2. ,  2.4,  2.8,  3.2,  3.6,  4. ,
        4.4,  4.8,  5.2,  5.6,  6. ,  6.4,  6.8,  7.2,  7.6,  8. ,  8.4,
        8.8,  9.2,  9.6, 10. , 10.4, 10.8, 11.2, 11.6, 12. , 12.4, 12.8,
       13.2, 13.6, 14. , 14.4, 14.8, 15.2, 15.6, 16. , 16.4, 16.8, 17.2,
       17.6, 18. , 18.4, 18.8, 19.2, 19.6, 20.]

y_true = []
for x in x_values:
    y_true.append(math.pow(math.sin(x), 2))

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


