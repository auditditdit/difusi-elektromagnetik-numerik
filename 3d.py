import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D   # needed for 3D plotting

# ------------------------------------------------
# PARAMETER FISIKA
# ------------------------------------------------
mu = 1.0
sigma = 1.0
D = 0.005 / (mu * sigma)

# ------------------------------------------------
# DOMAIN 3D
# ------------------------------------------------
L = 1.0
nx = ny = nz = 64
dx = L / (nx - 1)
dy = dx
dz = dx

# Stability dt <= dx^2/(6D)
dt = 0.2 * dx**2 / (6*D)

x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
z = np.linspace(0, L, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# ------------------------------------------------
# INITIAL CONDITION — Gaussian sphere
# ------------------------------------------------
x0 = y0 = z0 = 0.5
sigma0 = 0.1
E = np.exp(-(((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2) / (2*sigma0**2)))

# ------------------------------------------------
# NUMERICAL STEP
# ------------------------------------------------
def step(E):
    E_new = E.copy()
    E_new[1:-1,1:-1,1:-1] = E[1:-1,1:-1,1:-1] + D*dt * (
        (E[2:,1:-1,1:-1] - 2*E[1:-1,1:-1,1:-1] + E[:-2,1:-1,1:-1]) / dx**2 +
        (E[1:-1,2:,1:-1] - 2*E[1:-1,1:-1,1:-1] + E[1:-1,:-2,1:-1]) / dy**2 +
        (E[1:-1,1:-1,2:] - 2*E[1:-1,1:-1,1:-1] + E[1:-1,1:-1,:-2]) / dz**2
    )
    # Boundary = 0
    E_new[0,:,:] = 0; E_new[-1,:,:]=0
    E_new[:,0,:] = 0; E_new[:,-1,:]=0
    E_new[:,:,0] = 0; E_new[:,:,-1]=0
    return E_new

# ------------------------------------------------
# REAL 3D SURFACE (slice at mid-z)
# ------------------------------------------------
z_slice = nz // 2
E_slice = E[:, :, z_slice]

# Plot setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X[:,:,0], Y[:,:,0], E_slice, cmap='viridis')
ax.set_zlim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("E")
ax.set_title("Electromagnetic Diffusion 3D Surface (Mid-Z Slice)")

# ------------------------------------------------
# ANIMATION
# ------------------------------------------------
t = 0.0
def update(frame):
    global E, t, surf

    # update field several steps per frame
    for _ in range(3):
        E = step(E)
        t += dt

    ax.clear()

    E_slice = E[:, :, z_slice]
    surf = ax.plot_surface(X[:,:,0], Y[:,:,0], E_slice, cmap="viridis")

    ax.set_zlim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("E")
    ax.set_title(f"Electromagnetic Diffusion 3D Surface\nz-slice at {z[z_slice]:.3f}, t={t:.4f}s")

    return surf,

ani = FuncAnimation(fig, update, frames=2000, interval=1, blit=False)
plt.show()