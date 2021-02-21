import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate

data = res

rows, cols = data.shape
X, Y = np.indices(data.shape)

xs = np.linspace(0, cols, 30)
ys = np.linspace(0, rows, 30)
X2, Y2 = np.meshgrid(xs, ys, indexing="ij")

# 2次元スプライン補間を行なう。
tck = interpolate.bisplrep(X, Y, data, s=0)
data2 = interpolate.bisplev(xs, ys, tck)

# 描画する。
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")
ax1.plot_surface(X, Y, data, cmap="magma", edgecolor="gray")
surf = ax2.plot_surface(X2, Y2, data2, cmap="magma", edgecolor="gray")
fig.colorbar(surf)

for ax in [ax1, ax2]:
    ax.set_zticks((10, 20, 30))
    ax.view_init(30, -10)

plt.show()
