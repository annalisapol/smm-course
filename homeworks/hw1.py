import numpy as np
import matplotlib.pyplot as plt


def l(theta):
    return (theta - 3) **2 + 1

def grad_l(theta):
    return 2 * (theta - 3)

def GD(l, grad_l, theta_0, eta, maxit):

    theta_vals = [theta_0]
    for k in range(maxit):
        theta = theta_0 - eta * grad_l(theta_0)
        theta_0 = theta

        theta_vals.append(theta)

    return theta, np.array(theta_vals)

theta_005, traj_005 = GD(l, grad_l, theta_0 = 0, eta = 0.05, maxit = 100)
theta_02, traj_02 = GD(l, grad_l, theta_0 = 0, eta = 0.2, maxit = 100)
theta_1, traj_1 = GD(l, grad_l, theta_0 = 0, eta = 1, maxit = 100)

etas = [0.05, 0.2, 1.0]

trajectories = {
    0.05: traj_005,
    0.2: traj_02,
    1.0: traj_1
}

fig, axes = plt.subplots(1, 3, figsize=(15, 3))
for i, (eta, thetas) in enumerate(trajectories.items()):
    ax = axes[i]
    sc = ax.scatter(thetas, np.zeros_like(thetas), c=range(len(thetas)), cmap='viridis')
    ax.set_title(f"Θ(k) on real line\nη = {eta}")
    ax.set_xlabel("θ")
    ax.set_yticks([])
    ax.grid(True)
    fig.colorbar(sc, ax=ax, label='Iteration k')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 3))
for i, (eta, thetas) in enumerate(trajectories.items()):
    ax = axes[i]
    L_vals = l(thetas)
    ax.plot(L_vals, marker='o')
    ax.set_title(f"L(Θ(k)) vs iteration\nη = {eta}")
    ax.set_xlabel("Iteration k")
    ax.set_ylabel("L(Θ(k))")
    ax.grid(True)

plt.tight_layout()
plt.show()