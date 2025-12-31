import numpy as np
import matplotlib.pyplot as plt

N = 200
X = np.linspace(0, 1, N)
Y = 2 * X + np.random.normal(0, 0.1, size=N)

def f(theta, X):
    return theta[0] + theta[1] * X

def l(theta, X, Y):
    r = f(theta, X) - Y
    return np.mean(r**2)

def grad_l(theta, X, Y):
    r = f(theta, X) - Y
    g0 = 2 * np.mean(r)
    g1 = 2 * np.mean(r * X)
    return np.array([g0, g1])

def GD(l, grad_l, X, Y, theta_0, eta, maxit):
    theta_vals = [theta_0]
    for k in range(maxit):
        theta = theta_0 - eta * grad_l(theta_0, X, Y)
        theta_0 = theta
        theta_vals.append(theta)
    return theta_0, np.array(theta_vals)


def SGD(l, grad_l, X, Y, Theta0, lr=1e-2, batch_size=32, epochs=10):
    
    Theta = Theta0.copy()

    Theta_hist = [Theta0].copy()
    loss_hist = [l(Theta, X, Y)]

    for epoch in range(epochs):
        shuffle_idx = np.arange(len(X))
        np.random.shuffle(shuffle_idx)

        Xs = X[shuffle_idx]
        Ys = Y[shuffle_idx]

        n_batches = len(Xs) // batch_size
        for batch in range(n_batches):

            Xb = Xs[batch * batch_size : (batch+1) * batch_size]
            Yb = Ys[batch * batch_size : (batch+1) * batch_size]

            g = grad_l(Theta, Xb, Yb)
            Theta = Theta - lr * g

            Theta_hist.append(Theta.copy())

        loss_hist.append(l(Theta, X, Y))
    return Theta, np.array(Theta_hist), np.array(loss_hist)

theta0 = np.array([0.0, 0.0])
epochs = 30
lr = 1e-2
batch_sizes = [1, 10, 50, 200]

results = {}

for B in batch_sizes:
    Theta_final, Theta_hist, loss_hist = SGD(l, grad_l, X, Y, theta0, lr=lr, batch_size=B, epochs=epochs)
    results[B] = {"Theta_hist": Theta_hist, "loss_hist": loss_hist, "Theta_final": Theta_final}


plt.figure(figsize=(7,4))
for B in batch_sizes:
    plt.plot(results[B]["loss_hist"], marker="x", label=f"B={B}")
#plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss L(Θ)")
plt.grid(alpha=0.3)
plt.title("Loss vs epoch for different batch sizes")
plt.legend()
plt.show()

plt.figure(figsize=(6,6))

for B in batch_sizes:
    th = results[B]["Theta_hist"]

    #  most noisy traj
    if B == 1:
        th_plot = th[::10]   # keep every 10th point
        lw = 1.0
        alpha = 0.7
    else:
        th_plot = th
        lw = 2.0
        alpha = 1.0

    plt.plot(
        th_plot[:,0],
        th_plot[:,1],
        linewidth=lw,
        alpha=alpha,
        label=f"B={B}"
    )

    # Start o
    plt.scatter(
        th_plot[0,0],
        th_plot[0,1],
        s=40,
        marker="o",
        color=plt.gca().lines[-1].get_color()
    )

    # End x
    plt.scatter(
        th_plot[-1,0],
        th_plot[-1,1],
        s=80,
        marker="X",
        color=plt.gca().lines[-1].get_color()
    )

plt.xlabel(r"$\Theta_0$")
plt.ylabel(r"$\Theta_1$")
plt.title(r"Trajectory in parameter space $(\Theta_0,\Theta_1)$")
plt.grid(alpha=0.3)
plt.legend()
plt.axis("equal") 
plt.show()

