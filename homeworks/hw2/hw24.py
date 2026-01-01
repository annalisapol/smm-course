import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("homeworks/data/insurance.csv")
features = ["age", "bmi", "children"] 
X = data[features].values
y = data["charges"].values.reshape(-1, 1)

X = (X - X.mean(axis=0)) / X.std(axis=0)
y = (y - y.mean()) / y.std()

X = np.hstack([np.ones((X.shape[0], 1)), X])
N, d = X.shape

def l(Theta, X, y):
    return np.mean((X @ Theta - y)**2)

def grad_l(Theta, X, y):
    r = X @ Theta - y 
    return (2 / X.shape[0]) * (X.T @ r)

def GD(l, grad_l, X, Y, theta_0, eta, maxit):
    Theta = theta_0.copy()

    Theta_hist = [Theta.copy()]
    loss_hist = [l(Theta, X, Y)]
    gradnorm_hist = [np.linalg.norm(grad_l(Theta, X, Y))]

    for _ in range(maxit):
        Theta = Theta - eta * grad_l(Theta, X, Y)

        Theta_hist.append(Theta.copy())
        loss_hist.append(l(Theta, X, Y))
        gradnorm_hist.append(np.linalg.norm(grad_l(Theta, X, Y)))

    return Theta, np.array(Theta_hist), np.array(loss_hist), np.array(gradnorm_hist)


def SGD(l, grad_l, X, Y, Theta0, lr=1e-2, batch_size=32, epochs=10, seed = 0):
    rng = np.random.default_rng(seed)
    Theta = Theta0.copy()

    loss_hist = [l(Theta, X, Y)]
    gradnorm_hist = [np.linalg.norm(grad_l(Theta, X, Y))]

    N = X.shape[0]

    for epoch in range(epochs):
        idx = rng.permutation(N)
        Xs, Ys = X[idx], Y[idx]

        n_batches = N // batch_size
        for b in range(n_batches):
            Xb = Xs[b * batch_size:(b + 1) * batch_size]
            Yb = Ys[b * batch_size:(b + 1) * batch_size]

            Theta = Theta - lr * grad_l(Theta, Xb, Yb)

        loss_hist.append(l(Theta, X, Y))
        gradnorm_hist.append(np.linalg.norm(grad_l(Theta, X, Y)))

    return Theta, np.array(loss_hist), np.array(gradnorm_hist)

theta0 = np.zeros((d,1))
batch_sizes = [1, 10, 50]
lr = 1e-2
iter = 100

theta_final_GD, theta_hist_GD, loss_GD, gradnorm_GD = GD(l, grad_l, X, y, theta0, eta=lr, maxit=iter)

theta_final_SGD = {}
loss_SGD = {}
gradnorm_SGD = {}

for b in batch_sizes:
    theta_final_SGD[b], loss_SGD[b], gradnorm_SGD[b] = SGD(
        l, grad_l, X, y, theta0, lr=lr, batch_size=b, epochs=iter, seed=0
    )

plt.figure(figsize = (7,4))
plt.plot(loss_GD, label="GD", linewidth=2)
for b in batch_sizes:
    plt.plot(loss_SGD[b], label=f"SGD B={b}", linewidth=2)
plt.xlabel("Epoch/Iter")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.title("Loss vs iteration for GD and SGD")
plt.show()

plt.figure(figsize=(7,4))
plt.plot(gradnorm_GD, label="GD")
for b in batch_sizes:
    plt.plot(gradnorm_SGD[b], label=f"SGD B={b}")
plt.xlabel("Epoch")
plt.ylabel("Full gradient norm")
plt.grid(alpha=0.3)
plt.legend()
plt.title("Full gradient norm vs epoch")
plt.show()

print("Final theta GD:\n", theta_final_GD.ravel())

for b in batch_sizes:
    print(f"Final theta SGD (B={b}):\n", theta_final_SGD[b].ravel())
