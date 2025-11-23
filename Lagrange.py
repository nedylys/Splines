#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from splinesH import PolygonAcquisition
minmax = 7
fig,ax1 = plt.subplots(1, 1, figsize=(8,10))
ax1.set_xlim((-minmax,minmax))
ax1.set_ylim((-minmax,minmax))
ax1.set_xlabel('x-axis')
ax1.set_ylabel('y-axis')
ax1.set_title("Interpolation de Lagrange") 
ax1.grid(True)


def Aitken_Neville(P, N, t):
    N = N - 1
    p = [[None for j in range(N + 1)] for i in range(N + 1)]
    for i in range(N + 1):
        p[0][i] = P[i]
    for k in range(1, N + 1):
        for i in range(N - k + 1):
            p[k][i] = (i + k - t) / k * p[k - 1][i] + (t - i) / k * p[k - 1][i + 1]
    return p[N][0]


def main():

    xp,yp = PolygonAcquisition(ax1,'ob','--b')
    n_points = len(xp)
    Points = [np.array([xp[i], yp[i]]) for i in range(n_points)]
   
    T = np.linspace(0, n_points - 1, 1000)
    x, y = [], []
    for t in T:
        xp1, yp1 = Aitken_Neville(Points, n_points, t)
        x.append(xp1)
        y.append(yp1)

    # === Amélioration du style de tracé ===

    ax1.plot(x,y,'r')

    plt.draw()
    plt.tight_layout()
    plt.show()
   
   
    #plt.title("Interpolation de Lagrange via le schéma d’Aitken–Neville", fontsize=13, fontweight='bold')
    #plt.xlabel("x", fontsize=12)
    #plt.ylabel("y", fontsize=12)
    #plt.legend(loc='best')
    #plt.grid(True, linestyle='--', alpha=0.6)
    #plt.tight_layout()
    #plt.show()

main()
