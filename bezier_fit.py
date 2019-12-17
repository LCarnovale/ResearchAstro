import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

def A(row, col, size):
    """Get the value of the element in a 
    lower triangular coeffecient matrix
    for a bezier curve of order `size-1`, ie
    for a matrix of size `size`."""
    return (-1)**(row - col) * (
        comb(row, col) * comb(size - 1, row)
    )

# Points:
P = np.array([
    [1, 1],
    [3, 1.5],
    [4, 4],
    [2, 3],
    [1, 2.6],
    [5, 7.6]
])

# Order:
n = len(P)

M = np.array([
    [A(row, col, n) for col in range(n)]
    for row in range(n)
])


# T = lambda t : np.array([[t**i] for i in range(n)])
def T(t):
    a = np.array([t**i for i in range(n)]).T
    size = np.size(t)
    if size > 1:
        return a.reshape(size, -1, 1)
    else:
        return a.reshape(-1, 1)


t_ax = np.linspace(0, 1, 100)

eval_curv = lambda x: (T(t_ax))

curve = (T(t_ax) * M).dot(P).sum(axis=1)

plt.plot(*P.T, "ro-")
plt.plot(*curve.T)

plt.show()

print("Fitting to a parabola...")

data = np.array([[1, 2, 3, 4, 5], np.polyval([1, 1, 1], [1, 2, 3, 4, 5])]).T
diffs = data[1:] - data[:-1]
dists = np.linalg.norm(diffs, 2, 1)
s = np.array([0, *np.cumsum(dists)])
s /= s[-1]

inv = np.linalg.inv

Ts = T(s).reshape(-1, n)
Mi = inv(M)

C = np.matmul(
    np.matmul(Mi, inv(np.matmul(Ts.T, Ts))), 
    np.matmul(Ts.T, P[:-1])
)

plt.plot(*data.T)
plt.plot(*C.T, "ro")
plt.show()
