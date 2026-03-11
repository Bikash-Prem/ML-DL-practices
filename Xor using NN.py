import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]]
                 )

y = torch.tensor([[0.], [1.], [1.], [0.]])

torch.manual_seed(1)

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(5000):
    optimizer.zero_grad()

    output = model(x)
    loss = criterion(output, y)

    loss.backward()
    optimizer.step()

print(model(x))
