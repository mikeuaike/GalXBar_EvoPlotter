import glob
import numpy as np
import unsio.input as uns_in
import matplotlib.pyplot as plt


# Function BARSTRENGTH definition
def barstrength(mass, x, y, nbins=30, Rmin=0.0, Rmax=10.0, overf=1.0):
    '''
    Function to compute bar strength

    The amplitude of the m=2 Fourier component of the mass distribution
    (relative to the m=0) is computed as a function of radius. The quantity
    A2 is its maximum value within the specified radial range.

    Input: arrays of masses and positions

    Optional keywords:
        nbins : number of rings (default 30)
        Rmin  : minimum radius (default 0.0 kpc)
        Rmax  : maximum radius (default 10.0 kpc)
        overf : oversampling factor for rings (default 1.0)

    Returns: the value of A2
    '''

    R = np.sqrt(x**2+y**2)
    dR = (Rmax-Rmin) / nbins
    over = overf * dR

    I2_array = np.zeros(nbins)
    R_array = np.zeros(nbins)

    for i in range(0, nbins):

        R1 = Rmin + i*dR
        R2 = R1 + dR
        cond = np.argwhere((R > R1-over) & (R <= R2+over)).flatten()
        xcut = x[cond]
        ycut = y[cond]
        mcut = mass[cond]

        phi = np.arctan2(ycut, xcut)
        a0 = np.sum(mcut)
        a2 = np.sum(mcut * np.cos(2*phi))
        b2 = np.sum(mcut * np.sin(2*phi))
        I2 = np.sqrt(a2**2+b2**2) / a0

        I2_array[i] = I2
        R_array[i] = 0.5*(R1+R2)

    A2 = np.max(I2_array)

    return A2

# List and Sort files


names = sorted(glob.glob('./snapshots/snapshot*'))
n = len(names)
i2_array = np.zeros(n)
time_array = np.zeros(n)


for i in range(n):
    # read snapshot
    snapshotIn = names[i]
    s = uns_in.CUNS_IN(snapshotIn, 'all')
    s.nextFrame()

    _, pos = s.getData('disk', 'pos')
    _, mass = s.getData('disk', 'mass')
    _, time = s.getData('time')

    x = pos[0::3]
    y = pos[1::3]


# Find Center:
    x_com = np.sum(mass*x) / np.sum(mass)
    y_com = np.sum(mass*y) / np.sum(mass)

# Array Shift:
    x = x - x_com
    y = y - y_com

# Append
    time_array[i] = time
    i2_array[i] = barstrength(mass, x, y)

# Oldschool Print
a2 = np.argmax(i2_array)
print("TIME / I2 ")
for z in range(n):
    print(time_array[z], '-', i2_array[z])

print("\nA2: -------------------------- \n(Time): ", time_array[a2], '/', "(Value): ", i2_array[a2])

# Plot
label_0 = "i2 values"
x_axis = time_array
y_axis = i2_array
fig, ax = plt.subplots()
ax.scatter(x_axis, y_axis, label=label_0)

plt.title("Bar Strength through galaxy evolution")
plt.xlabel('Time [Gyr]')
plt.ylabel('I2')

plt.show()
