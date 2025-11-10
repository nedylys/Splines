#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from random import randint

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
    N = int(input("Entrer la valeur : "))
    # Points = [np.array([randint(0,10),randint(0,10)]) for _ in range(N)]
    Points = [
        np.array([0, 0]),
        np.array([1, np.sin(1)]),
        np.array([2, np.sin(2)]),
        np.array([3, np.sin(3)]),
        np.array([4, np.sin(4)]),
        np.array([5, np.sin(5)])
    ]

    T = np.linspace(0, N - 1, 1000)
    x, y = [], []
    for t in T:
        xp1, yp1 = Aitken_Neville(Points, N, t)
        x.append(xp1)
        y.append(yp1)

    # === Amélioration du style de tracé ===
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='dodgerblue', linewidth=2.2, label="Interpolation d'Aitken–Neville")
    plt.scatter([p[0] for p in Points], [p[1] for p in Points],
                color='crimson', s=60, marker='o', label='Points d\'interpolation')
    plt.title("Interpolation de Lagrange via le schéma d’Aitken–Neville", fontsize=13, fontweight='bold')
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

main()
