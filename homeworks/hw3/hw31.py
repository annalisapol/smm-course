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

plt.figure(figsize=(4, 4))

plt.scatter(X[y == 0, 0], X[y == 0, 1],
            color="tab:blue", alpha=0.7, label="Class 0")

plt.scatter(X[y == 1, 0], X[y == 1, 1],
            color="tab:orange", alpha=0.7, label="Class 1")

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Toy 2D Gaussian Dataset")
plt.legend()
plt.grid(alpha=0.3)
plt.axis("equal")

plt.show()

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

def GD(l, grad_l, Theta0, X, Y, eta, maxit):
    Theta = Theta0.copy()
    Theta_vals = [Theta.copy()]
    losses = [l(Theta, X, Y)]

    for k in range(maxit):
        Theta = Theta - eta * grad_l(Theta, X, Y)
        Theta_vals.append(Theta.copy())
        losses.append(l(Theta, X, Y))

    return Theta, np.array(Theta_vals), np.array(losses)
X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  
Y = y.reshape(-1, 1)                                

eta = 0.1
maxit = 1000
Theta0 = np.zeros((X_aug.shape[1], 1))              

Theta_star, Theta_hist, loss_hist = GD(
    l, grad_l,
    Theta0,
    X_aug, Y,
    eta=eta,
    maxit=maxit
)

b, w1, w2 = Theta_star.ravel()

plt.figure(figsize=(5, 5))

plt.scatter(X[y == 0, 0], X[y == 0, 1], alpha=0.7, label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], alpha=0.7, label="Class 1")

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x1_line = np.linspace(x1_min, x1_max, 200)

if abs(w2) > 1e-12:
    x2_line = -(b + w1 * x1_line) / w2
    plt.plot(x1_line, x2_line, linewidth=2, label=r"$\Theta^T x = 0$")
else:
    x1_boundary = -b / w1
    plt.axvline(x1_boundary, linewidth=2, label=r"$\Theta^T x = 0$")

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Learned decision boundary (logistic regression)")
plt.grid(alpha=0.3)
plt.axis("equal")
plt.legend()
plt.show()