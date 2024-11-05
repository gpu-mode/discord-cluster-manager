import torch
import torch.nn as nn
import torch.optim as optim

class NN(nn.Module):
    """
    Simple test network
    """
    def __init__(self):
        super(NN, self).__init__()
        self.fully_connected_layer = nn.Linear(10, 32)
        self.fully_connected_layer2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fully_connected_layer(x))
        return self.fully_connected_layer2(x)

def train(epochs: int = 5):
    x_train = torch.randn(1000, 10)
    y_train = torch.randn(1000, 1) 

    model = NN()
    costfunc = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=9e-1)

    for epoch in range(num_epochs):
        for i in range(0, len(x_train), 32):
            inputs = x_train[i:i+32]
            targets = y_train[i:i+32]

            outputs = model(inputs)
            loss = costfunc(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch+1} out of {epochs}, loss: {loss.item()}")

if __name__ == "__main__":
    train()
