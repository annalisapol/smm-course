import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1, 0], [0, 25]])


def l(theta):
    return 0.5 * theta.T @ A @ theta


def grad_l(theta):
    return A @ theta


def GD(l, grad_l, theta_0, eta, maxit):

    theta_vals = [theta_0]
    for k in range(maxit):
        theta = theta_0 - eta * grad_l(theta_0)
        theta_0 = theta

        theta_vals.append(theta)

    return theta, np.array(theta_vals)


def quad_levelsets(A, xlim=(-3, 3), ylim=(-3, 3), ngrid=400, ncontours=12, title=None):
    xs = np.linspace(xlim[0], xlim[1], ngrid)
    ys = np.linspace(ylim[0], ylim[1], ngrid)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2)
    cs = plt.contour(X, Y, Z, levels=ncontours)
    plt.clabel(cs, inline=True, fontsize=8)
    plt.axhline(0, lw=0.5, color="k")
    plt.axvline(0, lw=0.5, color="k")
    plt.gca().set_aspect("equal", "box")

    if title:
        plt.title(title)

    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.grid(alpha=0.2)


theta0 = np.array([-1.0, 1.0])
final_theta, path = GD(l, grad_l, theta0, eta=0.02, maxit=50)

quad_levelsets(A, title="Level sets + GD path (η = 0.02)")
plt.plot(path[:, 0], path[:, 1], "o-", color="red")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()

final_theta, path = GD(l, grad_l, theta0, eta=0.05, maxit=50)

quad_levelsets(A, title="Level sets + GD path (η = 0.05)")
plt.plot(path[:, 0], path[:, 1], "o-", color="red")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()

final_theta, path = GD(l, grad_l, theta0, eta=0.1, maxit=50)

quad_levelsets(A, title="Level sets + GD path (η = 0.1)")
plt.plot(path[:, 0], path[:, 1], "o-", color="red")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.show()
