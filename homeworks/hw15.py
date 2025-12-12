import numpy as np
import matplotlib.pyplot as plt


def rosenbrock(theta):
    return (1 - theta[0]) ** 2 + 100 * (theta[1] - theta[0] ** 2) ** 2


def grad_rosenbrock(theta):
    return np.array(
        [
            400 * theta[0] ** 3 - 400 * theta[0] * theta[1] + 2 * theta[0] - 2,
            200 * (theta[1] - theta[0] ** 2),
        ]
    )


def GD(l, grad_l, theta_0, eta, maxit):

    theta_vals = [theta_0]
    for k in range(maxit):
        theta = theta_0 - eta * grad_l(theta_0)
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

        if (np.linalg.norm(grad_l(theta)) < tolL) or (
            np.linalg.norm(theta - theta_0) < toltheta
        ):
            break

        theta_0 = theta
    return theta, np.array(thetas), k

def rosenbrock_levelsets(xlim=(-2, 2), ylim=(-1, 3), ngrid=400, ncontours=40, title=None):
    xs = np.linspace(xlim[0], xlim[1], ngrid)
    ys = np.linspace(ylim[0], ylim[1], ngrid)

    X, Y = np.meshgrid(xs, ys)

    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    cs = plt.contour(X, Y, Z, levels=ncontours,  cmap='jet')
    plt.clabel(cs, inline=True, fontsize=8)
    plt.axhline(0, lw=0.5, color="k")
    plt.axvline(0, lw=0.5, color="k")
    plt.gca().set_aspect("equal", "box")

    if title:
        plt.title(title)

    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.grid(alpha=0.2)


thetas_0 = [
    np.array([-1.5, 2]),
    np.array([-1, 0]),
    np.array([0, 2]),
    np.array([1.5, 1.5]),
]

etas = [1e-3, 1e-4, 1e-5]

for theta0 in thetas_0:

    gd_paths = {}
    gd_losses = {}

    for eta in etas:
        _, path_gd = GD(rosenbrock, grad_rosenbrock, theta0, eta=eta, maxit=20000)
        gd_paths[eta] = path_gd
        gd_losses[eta] = [rosenbrock(t) for t in path_gd]

    _, path_bt, _ = GD_backtracking(
        rosenbrock, grad_rosenbrock, theta0, maxit=50000, tolL=1e-6, toltheta=1e-10
    )
    loss_bt = [rosenbrock(t) for t in path_bt]

    methods = [
        ("GD η=1e-3", gd_paths[etas[0]], gd_losses[etas[0]]),
        ("GD η=1e-4", gd_paths[etas[1]], gd_losses[etas[1]]),
        ("GD η=1e-5", gd_paths[etas[2]], gd_losses[etas[2]]),
        ("Backtracking", path_bt, loss_bt),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 6), constrained_layout=True)


    for col, (title, traj, losses) in enumerate(methods):

        ax_traj = axes[0, col]

        xs = np.linspace(-2, 2, 300)
        ys = np.linspace(-1, 3, 300)
        X, Y = np.meshgrid(xs, ys)
        Z = (1 - X) ** 2 + 100 * (Y - X**2) ** 2

        levels = np.logspace(-1, 3, 25)
        ax_traj.contour(
            X, Y, Z,
            levels=levels,
            cmap="Greys",
            linewidths=0.6,
            alpha=0.6
        )

        ax_traj.plot(
            traj[:, 0], traj[:, 1],
            color="firebrick",
            linewidth=1.6,
            alpha=0.9
        )   

        ax_traj.scatter(traj[0, 0], traj[0, 1], s=30, color="firebrick", zorder=3)
        ax_traj.scatter(traj[-1, 0], traj[-1, 1], s=45, color="black", zorder=3)
        ax_traj.scatter([1], [1], marker="*", s=70, color="black", zorder=4)


        ax_traj.set_title(title, fontsize=11)
        ax_traj.set_aspect("equal")
        ax_traj.set_xticks([])
        ax_traj.set_yticks([])
        ax_traj.grid(alpha=0.15)

        ax_loss = axes[1, col]
        ax_loss.plot(
            losses,
            linewidth=1.6,
            color="firebrick",
            alpha=0.9
        )
        ax_loss.set_ylim(1e-12, 1e2)
        ax_loss.set_yscale("log")
        ax_loss.grid(alpha=0.3)

        if col == 0:
            ax_loss.set_ylabel("Loss (log)")
        ax_loss.set_xlabel("Iteration")

    fig.suptitle(r"$\theta_0$" + f" = {theta0}", fontsize=14)
    plt.show()