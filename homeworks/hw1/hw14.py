import numpy as np
import matplotlib.pyplot as plt

A = np.array([[5,0], [0,2]])

def l(theta):
    return 0.5 * theta.T @ A @ theta

def grad_l(theta):
    return A @ theta


def GD_exact_line_search(l, grad_l, theta_0, maxit):

    theta_vals = [theta_0]
    for k in range(maxit):
        g = grad_l(theta_0)
        num = g @ g
        denom = g @ A @ g
        eta = num / denom
        theta = theta_0 - eta * g
        theta_0 = theta

        theta_vals.append(theta)

    return theta, np.array(theta_vals)

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


theta0 = np.array([3.0, 3.0])

final_theta_exact, path_exact = GD_exact_line_search(l, grad_l, theta0, maxit=50)
final_theta_bt, path_bt, iter_bt = GD_backtracking(l, grad_l, theta0, maxit=50, tolL=1e-6, toltheta=1e-6)

quad_levelsets(A, title="Level sets + path - Exact Line Search vs Backtracking")
plt.plot(path_exact[:, 0], path_exact[:, 1], "o-", color="red", label="Exact Line Search")
plt.plot(path_bt[:, 0], path_bt[:, 1], "o-", color="blue", label="Backtracking")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend()
plt.show()

loss_exact =[l(theta) for theta in path_exact]
loss_bt = [l(theta) for theta in path_bt]

plt.figure(figsize = (6,4))
plt.plot(loss_exact, label = "Exact Line Search", marker = 'o')
plt.plot(loss_bt, label = "Backtracking", marker = 'o')
plt.yscale('log')
plt.xlabel("Iteration k")
plt.ylabel("Loss L(0(k))")
plt.title("Loss vs Iteration")
plt.grid(alpha=0.3)
plt.legend()
plt.show()
