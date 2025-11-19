import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

mu = 1.0
sigma = 1.0
D = 0.075 / (mu * sigma)

# domain
Lx = Ly = 1.0
nx = ny = 101
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)

# Stability: dt <= dx^2/(4D) for 2D
safety = 0.3
dt = safety * min(dx,dy)**2 / (4*D)

# jumlah langkah total simulasi
steps_per_cycle = 300   

# Initial: Gaussian bump
X, Y = np.meshgrid(x, y, indexing='ij')
x0, y0 = 0.4, 0.5
sigma0 = 0.05
E = np.exp(-(((X-x0)**2 + (Y-y0)**2) / (2*sigma0**2)))

def step(E):
    E_new = E.copy()
    E_new[1:-1,1:-1] = E[1:-1,1:-1] + D*dt*(
        (E[2:,1:-1] - 2*E[1:-1,1:-1] + E[:-2,1:-1]) / dx**2 +
        (E[1:-1,2:] - 2*E[1:-1,1:-1] + E[1:-1,:-2]) / dy**2
    )
    E_new[0,:] = 0
    E_new[-1,:] = 0
    E_new[:,0] = 0
    E_new[:,-1] = 0
    return E_new

# -----------------------------------------
# SIMULASI SEKALI 
# -----------------------------------------
E_history = []
E_now = E.copy()
t_history = []

t = 0.0
for i in range(steps_per_cycle):
    E_now = step(E_now)
    t += dt
    E_history.append(E_now.copy())
    t_history.append(t)

# -----------------------------------------
# ANIMASI LOOPING
# -----------------------------------------
fig, ax = plt.subplots()
im = ax.imshow(E_history[0].T, origin='lower', extent=(0,Lx,0,Ly),
               vmin=0, vmax=1, cmap='viridis')

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
cbar = fig.colorbar(im, ax=ax)

time_text = ax.text(0.02, 0.95, f"t = 0.0000 s",
                    transform=ax.transAxes, color='white')

def update(frame):
    im.set_array(E_history[frame].T)
    time_text.set_text(f"t = {t_history[frame]:.4f} s")
    return im, time_text

ani = FuncAnimation(
    fig, update,
    frames=len(E_history),
    interval=40,
    blit=True,
    repeat=True  
)

plt.title('Difusi Elektromagnetik 2D (Looping)')
plt.show()