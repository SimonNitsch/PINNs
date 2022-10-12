import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation



def gaussian(x,x0,sigma):

    return (np.pi*sigma**2)**-0.25 * np.exp(-(x-x0)**2/2/sigma**2)

def potential(x,a,b,c):

    return a * (np.tanh(b*(x+c)) - np.tanh(b*(x-c)))/2


def finite_diff(lb,ub,tmax,xn,tn,h=1,m=1,a=5,b=200,c=0.2):
    
    x = np.linspace(lb,ub,xn)
    t = np.linspace(lb,ub,tn)

    xk = (ub-lb)/xn
    tk = tmax/tn

    psi = []
    psi.append(gaussian(x,lb+(ub-lb)/4,(ub-lb)/50))

    main_diag = 1j*h/m*tk/xk**2 + 1 + 1j*tk/h*potential(x,a,b,c)
    main_diag[main_diag <= 1e-14] = 1e-14
    side_diag = h/2/m*tk/xk**2 * np.ones(xn-1) * -1j
    
    A = np.diag(main_diag) + np.diag(side_diag,1) + np.diag(side_diag,-1)
    print("Matrix-Inversion")
    Ainv = np.linalg.inv(A)


    for k in range(0,tn):
        psi_last = psi[-1]
        psi_new = Ainv@psi_last
        psi.append(psi_new)
        print("Berechnet Zeitpunkt %s" % (k+1))
    
    print("Beendet")

    return np.stack(psi)



if __name__ == "__main__":

    lb = -5
    ub = 5
    a = 5
    c = 0.2


    psi = finite_diff(lb,ub,5,20000,1000,a=a,c=c)

    def psifunc(frame):
        line[0].set_data([np.linspace(-5,5,20000),np.abs(psi[frame,:])**2])
        return line

    frames = np.arange(1000)

    fig = plt.figure()
    ax = plt.axes(xlim=(-5,5),ylim=(0,1.5))
    line = ax.plot([],[])
    ani = animation.FuncAnimation(fig,psifunc,frames=frames,interval=20)
    point1 = ax.plot([-c,-c],[2,-2],color="r")
    point2 = ax.plot([c,c],[2,-2],color="r")
    plt.show()








