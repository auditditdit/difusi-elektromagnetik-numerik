# 1d py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# PARAMETER FISIKA
mu = 1.0
sigma = 1.0
D = 1.0 / (mu * sigma)

# DOMAIN
L = 1.0
nx = 200
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

# Kestabilan difusi
dt = 0.4 * dx**2 / (2 * D)

# Waktu simulasi
steps_per_cycle = 500

# INITIAL CONDITION: Gaussian
x0 = 0.3
sigma0 = 0.04
E0 = np.exp(-((x - x0)**2) / (2 * sigma0**2))

# Simpan riwayat untuk looping
E_history = []
E = E0.copy()

for i in range(steps_per_cycle):
    E_new = E.copy()
    E_new[1:-1] = E[1:-1] + D * dt / dx**2 * (E[2:] - 2*E[1:-1] + E[:-2])
    E_new[0] = 0
    E_new[-1] = 0
    E = E_new.copy()
    E_history.append(E.copy())

# ANIMASI LOOPING
fig, ax = plt.subplots()
line, = ax.plot(x, E0, lw=2)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("x (m)")
ax.set_ylabel("E(x,t)")
ax.set_title("Difusi Elektromagnetik 1D (E field)")

# Tambahkan timestamp
time_text = ax.text(0.02, 0.90, "", transform=ax.transAxes, fontsize=12)

def update(frame):
    line.set_ydata(E_history[frame])
    time_text.set_text(f"t = {frame * dt:.4f} s")
    return line, time_text

ani = FuncAnimation(
    fig, update,
    frames=len(E_history),
    interval=30,
    blit=True,
    repeat=True
)

plt.show()