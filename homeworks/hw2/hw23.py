import numpy as np
import matplotlib.pyplot as plt
def l(theta):
    return (theta[0]**2 - 1)**2 + 10*(theta[1] - theta[0]**2)**2

def grad_l(theta):
    theta1, theta2 = theta

    dtheta1 = (
        4 * theta1 * (theta1**2 - 1)
        - 40 * theta1 * (theta2 - theta1**2)
    )

    dtheta2 = 20 * (theta2 - theta1**2)

    return np.array([dtheta1, dtheta2])

def SGD_noisy(grad_l, theta0, lr=1e-2, sigma2 = 0.001, n_steps = 5000, seed=0):
    rng = np.random.default_rng(seed)

    theta = theta0.copy()
    theta_hist = [theta0.copy()]

    for k in range(n_steps):
        grad = grad_l(theta)
        noise = rng.normal(0.0, np.sqrt(sigma2), size = 2)
        gk = grad + noise

        theta = theta - lr*gk
        theta_hist.append(theta.copy())
    
    return np.array(theta_hist)

sigma2_vals = [0.0, 0.01, 0.1]
theta0 = np.array([-1.2, 1.6])
lrs = [1e-4, 3e-4, 1e-3]
T = 1000
results = {}

for lr in lrs:
    for sigma2 in sigma2_vals:
        results[(lr, sigma2)] = SGD_noisy(
            grad_l, theta0, lr=lr, sigma2=sigma2, n_steps=5000, seed=42
        )

def quad_levelsets(xlim=(-3.0, 3.0), ylim=(-1.0, 4.0), ngrid=400, levels = None, title=None):
    xs = np.linspace(xlim[0], xlim[1], ngrid)
    ys = np.linspace(ylim[0], ylim[1], ngrid)
    X, Y = np.meshgrid(xs, ys)
    Z = (X**2 - 1)**2 + 10 * (Y - X**2)**2

    
    plt.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.85)
    cs = plt.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.5)

    plt.clabel(cs, inline=True, fontsize=8)
    plt.axhline(0, lw=0.5, color="k")
    plt.axvline(0, lw=0.5, color="k")

    if title:
        plt.title(title)

    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.grid(alpha=0.2)
levels = np.logspace(-2, 3, 15)

for lr in lrs:
    plt.figure(figsize=(7,5))

    quad_levelsets(
        xlim=(-3, 3),
        ylim=(-0.5, 2.5),
        ngrid=500,
        levels=levels,
        title=fr"Trajectories on level sets (lr={lr}, first {T} steps)"
    )

    for sigma2 in sigma2_vals:
        th = results[(lr, sigma2)][:T]  
        plt.plot(th[:,0], th[:,1], lw=2, label=fr"$\sigma^2={sigma2}$")
        plt.plot(th[0,0], th[0,1], "o", ms=5)     
        plt.plot(th[-1,0], th[-1,1], "x", ms=7)   

    plt.scatter([1, -1], [1, 1], marker="*", s=120, color="red", zorder=5)

    plt.legend()
    plt.show()
