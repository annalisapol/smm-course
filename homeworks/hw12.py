import numpy as np
import matplotlib.pyplot as plt

def l(theta):
    return theta**4 - 3*(theta**2) + 2
def grad_l(theta):
    return 4*(theta**3) - 6*(theta)

def backtracking(L, grad_L, theta, eta0=1.0, beta=0.5, c=1e-4):
    eta = eta0
    g = grad_L(theta)
    g_norm2 = np.dot(g, g)
    while L(theta - eta * g) > L(theta) - c * eta * g_norm2:
        eta *= beta
    return eta

def GD_backtracking(l, grad_l, theta_0, maxit, tolL, toltheta):
    thetas = [theta_0] 
    for k in range(maxit):
        eta = backtracking(l, grad_l, theta_0)
        theta = theta_0 - eta * grad_l(theta_0)
        thetas.append(theta)

        if (np.linalg.norm(grad_l(theta)) < tolL) or (np.linalg.norm(theta - theta_0) < toltheta):
            break

        theta_0 = theta
    return theta, np.array(thetas), k

thetas_0 = [-2, 0.5, 2]
results = {}

for theta_0 in thetas_0:
    theta_final, traj, k = GD_backtracking(l, grad_l, theta_0 = theta_0, maxit = 1000, tolL = 1e-6, toltheta = 1e-6)
    results[theta_0] = {"final": theta_final, "trajectory": traj, "iters": k}
    print(f"Theta₀ = {theta_0}: converged to θ* = {theta_final:.5f} in {k} iterations")


theta_vals = np.linspace(-3, 3, 400)
L_vals = l(theta_vals)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
colors = ['blue', 'orange', 'green']

for i, theta_0 in enumerate(thetas_0):
    ax = axes[i]
    traj = results[theta_0]["trajectory"]

    ax.plot(theta_vals, L_vals, 'k-', label='L(θ)')

    ax.scatter(traj, l(traj), color=colors[i], alpha=0.7, s=50, edgecolors='k', label='Iterates')

    ax.set_title(f"Start θ₀ = {theta_0}")
    ax.set_xlabel("θ")
    if i == 0:
        ax.set_ylabel("L(θ)")
    ax.legend()
    ax.grid(True)

plt.suptitle("Gradient Descent with Backtracking – trajectories from different initial points", fontsize=13)
plt.tight_layout()
plt.show()
