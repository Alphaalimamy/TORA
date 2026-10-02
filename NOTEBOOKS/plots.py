def plot():
    import torch
    import numpy as np
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    # 1. Data
    np.random.seed(42)
    torch.manual_seed(42)

    x_train = np.random.rand(100, 1).astype(np.float32)
    y_train = 2 * x_train + 1 + 0.1 * np.random.randn(100, 1).astype(np.float32)

    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)

    #  2. Model
    class ManualLinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(1))
            self.b = nn.Parameter(torch.randn(1))
        def forward(self, x):
            return self.w * x + self.b

    model = ManualLinearRegression()
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=1e-1)

    # 3. Train
    n_epochs = 1000
    for epoch in range(n_epochs):
        model.train()
        yhat = model(x_train_tensor)
        loss = loss_fn(yhat, y_train_tensor)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 4. Plot
    # Scatter the raw data
    plt.scatter(x_train, y_train, s=12, alpha=0.6, label="data")

    # Compute the fitted line over a dense x-range
    with torch.no_grad():
        x_line = torch.linspace(0, 1, 100).reshape(-1, 1)
        y_line = model(x_line).numpy()

    plt.plot(x_line.numpy(), y_line, color="red", linewidth=2,
             label=f"fit: y = {model.w.item():.3f}x + {model.b.item():.3f}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear regression fit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == '__main__':
    plot()