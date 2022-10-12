import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation

X2 = np.load("psiTunnel.npy")

x = np.linspace(-12,8,750)
x2 = np.linspace(-5,5,20000)
t = np.ones((750)) * 5/1000
psi = tf.keras.models.load_model("GaussTunnel2.h5")

def psifunc(frame):

    X = np.vstack((x,t*frame)).T
    Y = psi(X)

    line[0].set_data([x,Y[:,0]**2+Y[:,1]**2])
    return line

frames = np.arange(1000)

fig = plt.figure()
ax = plt.axes(xlim=(-5,5),ylim=(0,1.5))
line = ax.plot([],[])
ani = animation.FuncAnimation(fig,psifunc,frames=frames,interval=5)
point1 = ax.plot([-0.2,-0.2],[2,-2],color="r")
point2 = ax.plot([0.2,0.2],[2,-2],color="r")
plt.show()


fig2 = plt.figure()
ax2 = fig2.add_subplot(xlim=(-5,5),ylim=(0,0.525))
X = np.vstack((x,t*300)).T
Y = psi(X)
line2 = ax2.plot(x,Y[:,0]**2+Y[:,1]**2)
line3 = ax2.plot(x2,abs(X2[300,:])**2,c="r")
ax2.set_ylabel(r"$|\psi(x,t\!=\!1,\!5)|^2$")
ax2.set_xlabel("x")
ax2.legend(["PINN","Implizite Finite-Differenzen"])
plt.show()


