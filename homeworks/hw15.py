import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(theta):
    return (1 - theta[0])**2 + 100*(theta[1] - theta[0]**2)**2

def grad_rosenbrock(theta):
    return np.array([400*theta[0]**3 - 400*theta[0]*theta[1] + 2*theta[0] - 2 , 200*(theta[1] - theta[0]**2)])

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

        if (np.linalg.norm(grad_l(theta)) < tolL) or (np.linalg.norm(theta - theta_0) < toltheta):
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

thetas_0 = [np.array([-1.5, 2]), np.array([-1, 0]), np.array([0, 2]), np.array([1.5, 1.5])]

etas = [1e-3, 1e-4, 1e-5] 

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
axes = axes.reshape(4,4)

for row, theta0 in enumerate(thetas_0):

    gd_paths = {}
    for eta in etas:
        theta_gd, path_gd = GD(rosenbrock, grad_rosenbrock,
                               theta0, eta=eta, maxit=20000)
        gd_paths[eta] = path_gd

    theta_bt, path_bt, it_bt = GD_backtracking(
        rosenbrock, grad_rosenbrock, theta0,
        maxit=50000, tolL=1e-6, toltheta=1e-10
    )

    method_list = [
        (f"GD η={etas[0]}", gd_paths[etas[0]]),
        (f"GD η={etas[1]}", gd_paths[etas[1]]),
        (f"GD η={etas[2]}", gd_paths[etas[2]]),
        ("Backtracking", path_bt)
    ]

    for col, (title, traj) in enumerate(method_list):

        ax = axes[row, col]

        xs = np.linspace(-2, 2, 250)
        ys = np.linspace(-1, 3, 250)
        X, Y = np.meshgrid(xs, ys)
        Z = (1 - X)**2 + 100*(Y - X**2)**2
        ax.contour(X, Y, Z, levels=20, cmap="Blues", linewidths=0.6, alpha=0.99)

        ax.plot(traj[:,0], traj[:,1], "-", linewidth=1.8, color="darkred")
        ax.scatter(traj[0,0], traj[0,1], s=20, color="red")     
        ax.scatter(traj[-1,0], traj[-1,1], s=35, color="black") 
        ax.scatter([1], [1], marker="*", color="black", s=60)    

        ax.set_title(f"{title}\ninit={theta0}", fontsize=9)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()
