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

batch_sizes = [1, 5, 20, N]

theta_fixed = np.array([0.0, 0.0])

def grad_samples_shuff(grad_l, X, Y, Theta_fixed, batch_size, K = 100, seed=0):
    rng = np.random.default_rng(seed)

    G = []

    while len(G) < K:
        shuffle_idx = np.arange(len(X))
        rng.shuffle(shuffle_idx)

        Xs = X[shuffle_idx]
        Ys = Y[shuffle_idx]

        n_batches = int(np.ceil(len(Xs) / batch_size))

        for b in range(n_batches):
            if len(G) >= K:
                break
                
            start = b * batch_size
            end = min((b+1)*batch_size, len(Xs))

            Xb = Xs[start:end]
            Yb = Ys[start:end]

            if (end - start) < batch_size:
                continue

            gk = grad_l(Theta_fixed, Xb, Yb)
            G.append(gk)
    
    return np.array(G)

grads_by_batch = {}

for b in batch_sizes:
    grads_by_batch[b] = grad_samples_shuff(grad_l, X, Y, theta_fixed, batch_size=b, K = 100, seed=42)

def empirical_variance(G):
    gbar = G.mean(axis = 0)
    diffs = G - gbar
    var = np.mean(np.sum(diffs**2, axis = 1))
    return var

vars_batch = {}

for b in batch_sizes:
    g = grads_by_batch[b]
    vars_batch[b] = empirical_variance(g)

b_arr = np.array(batch_sizes)
var_arr = np.array([vars_batch[b] for b in batch_sizes])

plt.figure(figsize=(6,4))
plt.plot(b_arr, var_arr, "o-")
plt.xlabel("Batch size B")
plt.ylabel(r"Empirical Var(g) (log scale)")
plt.grid(alpha=0.3, which="both")
plt.title("Variance of stochastic gradient vs batch size (fixed θ)")
plt.show()
