import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

from astropy.time import Time
from astroquery.jplhorizons import Horizons


# Conver .aei to csv file
def aei2csv(aeipath:str) -> None:
    with open(aeipath) as file:
        lines = file.readlines()

    csvfile = open(aeipath.replace(".aei", ".csv"), "w")

    for i, line in enumerate(lines):
        line = line.strip()
        while "  " in line:
            line = line.replace("  ", " ")
        
        line = line.replace("Time (years)", "Time")
        line = line.replace(" ", ",")
        
        if len(line) > 0 and i > 2:
            csvfile.write(line+"\n")

    print(aeipath.replace(".aei", ".csv"), "was successfully generated!")

    return None
            
    
aei2csv("./JUPITER.aei")
aei2csv("./HEKTOR.aei")

JUPITER = pd.read_csv("./JUPITER.csv")
HEKTOR = pd.read_csv("./HEKTOR.csv")

# In Jupier Reference Frame
Ex, Ey, Ez = JUPITER["x"].to_numpy(), JUPITER["y"].to_numpy(), JUPITER["z"].to_numpy()

fig, ax = plt.subplots(1,2, figsize = (10, 5))

ax[0].plot(HEKTOR["x"].to_numpy(), HEKTOR["y"].to_numpy(), "g+", label = "HEKTOR")
ax[0].plot(JUPITER["x"].to_numpy(), JUPITER["y"].to_numpy(), "bo")

ax[0].set_xlim(-10, 10)
ax[0].set_ylim(-10, 10)
ax[0].set_xlabel(r"$x_{\text{ICRF}}$ [au]")
ax[0].set_ylabel(r"$y_{\text{ICRF}}$ [au]")

ax[1].plot(HEKTOR["x"].to_numpy(), HEKTOR["z"].to_numpy(), "g+", label = "HEKTOR")
ax[1].plot(JUPITER["x"].to_numpy(), JUPITER["z"].to_numpy(), "bo")

ax[1].set_xlim(-10, 10)
ax[1].set_ylim(-10, 10)
ax[1].set_xlabel(r"$x_{\text{ICRF}}$ [au]")
ax[1].set_ylabel(r"$z_{\text{ICRF}}$ [au]")

ax[1].legend()

plt.savefig("orbit_plot.png", dpi=300, bbox_inches="tight")
plt.show()


def FrameConvert(input_target:pd.DataFrame, ref_target:pd.DataFrame)->tuple:
    Ir = input_target.loc[:, "x":"z"].to_numpy().T   # (3, N)
    Iv = input_target.loc[:, "vx":"vz"].to_numpy().T # (3, N)
    Rr = ref_target.loc[:, "x":"z"].to_numpy().T     # (3, N)
    Rv = ref_target.loc[:, "vx":"vz"].to_numpy().T   # (3, N)

    # X' Axis
    E1 = Rr / np.linalg.norm(Rr, axis=0)

    # Z' Axis // Orbital Angular Momentum Vector Direction
    Rh = np.cross(Rr, Rv, axis=0)
    E3 = Rh / np.linalg.norm(Rh, axis=0)

    # Y' Axis: orthogonal to X', Z' Axis
    E2 = np.cross(E3, E1, axis=0)

    # Rotation Matrix
    Rot = np.stack([E1.T, E2.T, E3.T], axis=1)

    Ir_new = np.einsum('nij, nj -> ni', Rot, Ir.T)
    Rr_new = np.einsum('nij, nj -> ni', Rot, Rr.T)

    Ir_rot = Ir_new.T
    Rr_rot = Rr_new.T

    return Ir_rot, Rr_rot


HEKTOR_rot, JUPITER_rot = FrameConvert(input_target=HEKTOR, ref_target=JUPITER)

fig, ax = plt.subplots(1,2, figsize = (10, 5))

ax[0].plot(HEKTOR_rot[0], HEKTOR_rot[1], "g+", label = "HEKTOR")
ax[0].plot(JUPITER_rot[0], JUPITER_rot[1], "bo")

ax[0].set_xlim(-6, 6)
ax[0].set_ylim(-6, 6)
ax[0].set_xlabel(r"$x_{\text{Jupier}}$ [au]")
ax[0].set_ylabel(r"$y_{\text{Jupier}}$ [au]")

ax[1].plot(HEKTOR_rot[0], HEKTOR_rot[2], "g+", label = "HEKTOR")
ax[1].plot(JUPITER_rot[0], JUPITER_rot[2], "bo")

ax[1].set_xlim(-6, 6)
ax[1].set_ylim(-6, 6)
ax[1].set_xlabel(r"$x_{\text{Jupier}}$ [au]")
ax[1].set_ylabel(r"$z_{\text{Jupier}}$ [au]")

ax[1].legend()

plt.savefig("orbit_fixed_plot.png", dpi=300, bbox_inches="tight")
plt.show()





try:
    os.mkdir("visualization", exist_ok=True)
except:
    print("Folder visualization already exists.")

HEKTOR_rot, JUPITER_rot = FrameConvert(input_target=HEKTOR, ref_target=JUPITER)
T = JUPITER["Time"].to_numpy()

N = len(HEKTOR_rot[0])

fig, ax = plt.subplots(1,2, figsize = (10, 5))
for i in range(N):
    ax[0].plot(HEKTOR_rot[0][i], HEKTOR_rot[1][i], "g+", label = "HEKTOR")
    ax[0].plot(JUPITER_rot[0][i], JUPITER_rot[1][i], "bo")

    ax[0].set_xlim(-6, 6)
    ax[0].set_ylim(-6, 6)
    ax[0].set_xlabel(r"$x_{\text{Jupiter}}$ [au]")
    ax[0].set_ylabel(r"$y_{\text{Jupiter}}$ [au]")

    ax[1].plot(HEKTOR_rot[0][i], HEKTOR_rot[2][i], "g+", label = "HEKTOR")
    ax[1].plot(JUPITER_rot[0][i], JUPITER_rot[2][i], "bo")

    ax[1].set_xlim(-6, 6)
    ax[1].set_ylim(-6, 6)
    ax[1].set_xlabel(r"$x_{\text{Jupiter}}$ [au]")
    ax[1].set_ylabel(r"$z_{\text{Jupiter}}$ [au]")

    ax[1].legend()

    ax[0].grid()
    ax[1].grid()

    ax[0].annotate(f"{int(T[i]):04d} yrs", (-2.2, 2.1), fontsize = 13)

    plt.savefig(f"visualization/Plot_{i+1:05d}.png")

    ax[0].clear()
    ax[1].clear()
    
plt.close(fig)


image_frames = sorted(glob('visualization/Plot_*.png'))

images = []
for filename in image_frames:
    img = Image.open(filename)
    images.append(img)

images[0].save('animation.gif',
                  save_all=True,
                  append_images=images[1:],
                  duration=150,
                  loop=0,
                  format='GIF')


plt.figure(figsize = (6,6))
plt.plot(JUPITER["x"], JUPITER["y"], "bo", label = "Earth")
plt.plot(HEKTOR["x"], HEKTOR["y"], "go", label="3I/HEKTOR (Mercury6)")
plt.xlim([-10, 10])
plt.ylim([-10, 10])
jd_time = Time("2025-07-01 00:00:00").jd
obj = Horizons(id="DES= 20000624", location="@sun", epochs=np.linspace(jd_time, jd_time+2000, 20))
vec = obj.vectors()

plt.plot(vec["x"], vec["y"], "r+", label="624/HEKTOR (JPL Horizons)")
plt.legend()
plt.xlabel(r"$x_{\text{ICRF}}$ [au]")
plt.ylabel(r"$y_{\text{ICRF}}$ [au]")

plt.savefig("comparison.png", dpi=300, bbox_inches="tight")
plt.show()
