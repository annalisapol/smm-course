import numpy as np
import matplotlib.pyplot as plt

n0 = 200
n1 = 200

mu0 = np.array([-2, -2])
mu1 = np.array([2, 2])

sigma0 = 1 * np.eye(2)
sigma1 = 0.5 * np.eye(2)

X0 = np.random.multivariate_normal(mu0, sigma0, n0)
X1 = np.random.multivariate_normal(mu1, sigma1, n1)

y0 = np.zeros(n0)
y1 = np.ones(n1)

X = np.vstack([X0, X1])
y = np.concatenate([y0, y1])

def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def f(Theta, X):
    return sigmoid(X @ Theta)

def l(Theta, X, Y):
    Y_hat = f(Theta, X)
    eps = 1e-12
    Y_hat = np.clip(Y_hat, eps, 1 - eps)
    return -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

def grad_l(Theta, X, Y):
    N = X.shape[0]
    Y_hat = f(Theta, X)
    return (X.T @ (Y_hat - Y)) / N

def accuracy(Theta, X, Y):
    p = f(Theta, X)
    y_pred = (p >= 0.5).astype(int)
    return np.mean(y_pred == Y)


def SGD(l, grad_l, X, Y, Theta0, lr=1e-2, batch_size=32, epochs=10):
    Theta = Theta0.copy()

    loss_hist = []
    acc_hist = []

    N = X.shape[0]

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        Xs, Ys = X[idx], Y[idx]

        for start in range(0, N, batch_size):
            Xb = Xs[start:start+batch_size]
            Yb = Ys[start:start+batch_size]

            g = grad_l(Theta, Xb, Yb)
            Theta = Theta - lr * g

        loss_hist.append(l(Theta, X, Y))
        acc_hist.append(accuracy(Theta, X, Y))

    return Theta, np.array(loss_hist), np.array(acc_hist)

X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
Y = y.reshape(-1, 1)

N = X_aug.shape[0]
batch_sizes = [1, 10, N]

lr = 0.1
epochs = 100
Theta0 = np.zeros((X_aug.shape[1], 1))

results = {}

for B in batch_sizes:
    Theta_star, loss_hist, acc_hist = SGD(
        l, grad_l,
        X_aug, Y,
        Theta0,
        lr=lr,
        batch_size=B,
        epochs=epochs
    )
    results[B] = {
        "Theta": Theta_star,
        "loss": loss_hist,
        "acc": acc_hist
    }
    print(f"B={B}: final loss={loss_hist[-1]:.4f}, final acc={acc_hist[-1]:.3f}")

plt.figure(figsize=(6,4))
for B in batch_sizes:
    plt.plot(results[B]["loss"], label=f"B={B}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch (SGD, different batch sizes)")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
for B in batch_sizes:
    plt.plot(results[B]["acc"], marker="o", markersize=3, label=f"B={B}")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch (SGD, different batch sizes)")
plt.grid(alpha=0.3)
plt.legend()
plt.ylim(0.99, 1.01)   
plt.show()