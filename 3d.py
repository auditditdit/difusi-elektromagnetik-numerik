import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mu = 1.0
sigma = 1.0
D = 1.0 / (mu * sigma)

# domain
L = 1.0
nx = ny = nz = 64   # timestep dan memori berlipat - kecilkan jika perlu
dx = L / (nx - 1)
dy = dx
dz = dx

# Stability approx for 3D explicit: dt <= dx^2/(6D)
safety = 0.2
dt = safety * dx**2 / (6*D)
t_final = 0.1
nt = int(t_final / dt)

x = np.linspace(0, L, nx)
y = np.linspace(0, L, ny)
z = np.linspace(0, L, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# initial: 3D gaussian sphere
x0 = y0 = z0 = 0.5*L
sigma0 = 0.08
E = np.exp(-(((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)/(2*sigma0**2)))
E_new = E.copy()

def step(E):
    E_new = E.copy()
    # update interior points
    E_new[1:-1,1:-1,1:-1] = E[1:-1,1:-1,1:-1] + D*dt * (
        (E[2:,1:-1,1:-1] - 2*E[1:-1,1:-1,1:-1] + E[:-2,1:-1,1:-1]) / dx**2 +
        (E[1:-1,2:,1:-1] - 2*E[1:-1,1:-1,1:-1] + E[1:-1,:-2,1:-1]) / dy**2 +
        (E[1:-1,1:-1,2:] - 2*E[1:-1,1:-1,1:-1] + E[1:-1,1:-1,:-2]) / dz**2
    )
    # boundary dirichlet 0
    E_new[0,:,:] = 0; E_new[-1,:,:]=0
    E_new[:,0,:] = 0; E_new[:,-1,:]=0
    E_new[:,:,0] = 0; E_new[:,:,-1]=0
    return E_new

# Setup figure: show 3 slices (z=lower, mid, upper)
fig, axes = plt.subplots(1,3, figsize=(12,4))
slice_idx = [int(nz*0.25), int(nz*0.5), int(nz*0.75)]
ims = []
vmin, vmax = 0.0, 1.0

for ax, k in zip(axes, slice_idx):
    im = ax.imshow(E[:,:,k].T, origin='lower', extent=(0,L,0,L), vmin=vmin, vmax=vmax)
    ax.set_title(f'z slice k={k}')
    ims.append(im)
fig.colorbar(ims[1], ax=axes, orientation='vertical')
time_text = fig.text(0.02, 0.95, '')

t = 0.0
def update(frame):
    global E, t
    steps_per_frame = max(1, nt//80)
    for _ in range(steps_per_frame):
        E = step(E)
        t += dt
    for im, k in zip(ims, slice_idx):
        im.set_array(E[:,:,k].T)
    time_text.set_text(f't = {t:.4f} s')
    return (*ims,)

ani = FuncAnimation(fig, update, frames=300, blit=False, interval=50)
plt.suptitle('Difusi Elektromagnetik 3D (3 slices)')
plt.show()