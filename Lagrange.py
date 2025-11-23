#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from splinesH import PolygonAcquisition, ManualPointAcquisition # Assurez-vous que splinesH est accessible


def Aitken_Neville(P, N, t):
    """
    Algorithme d'Aitken-Neville pour l'interpolation polynomiale de Lagrange.
    Calcule le point P(t) = (x(t), y(t)) à partir des points de contrôle P.
    """
    N = N - 1
    p = [[None for j in range(N + 1)] for i in range(N + 1)]
    for i in range(N + 1):
        p[0][i] = P[i]
    for k in range(1, N + 1):
        for i in range(N - k + 1):
            p[k][i] = (i + k - t) / k * p[k - 1][i] + (t - i) / k * p[k - 1][i + 1]
    return p[N][0]


def get_point_function(points):
    """
    Crée et retourne une fonction lambda P(t) qui utilise Aitken-Neville
    pour évaluer la courbe à un paramètre t.
    """
    n_points = len(points)
    return lambda t: Aitken_Neville(points, n_points, t)


def calculate_first_derivative(P_t, t, h=1e-6):
    """
    Calcule numériquement la première dérivée P'(t) en utilisant une approximation numérique.
    """
    P_plus = P_t(t + h)
    P_minus = P_t(t - h)
    
    # Formule des différences centrales : P'(t) ≈ (P(t+h) - P(t-h)) / (2h) une erreur de O(h²)
    d1 = (P_plus - P_minus) / (2 * h)
    return d1


def calculate_second_derivative(P_t, t, h=1e-6):
    """
    Calcule numériquement la seconde dérivée P''(t) en appoximation numérique
    """
    P_plus = P_t(t + h)
    P_minus = P_t(t - h)
    P_current = P_t(t)
    
    # Formule : P''(t) ≈ (P(t+h) - 2P(t) + P(t-h)) / (h²)
    d2 = (P_plus - 2 * P_current + P_minus) / (h**2)
    return d2


def calculate_curvature_from_derivatives(d1, d2):
    """
    Calcule la courbure κ à partir des vecteurs de première (d1) 
    et seconde (d2) dérivée.
    """
    # κ = |x'y'' - y'x''| / ( (x'^2 + y'^2)^(3/2) )
    numerateur = d1[0] * d2[1] - d1[1] * d2[0]
    denominateur = np.linalg.norm(d1)**3
    
    # Évite la division par zéro si la vitesse est nulle
    kappa = numerateur / denominateur if denominateur > (1e-6)**3 else 0 
    
    return kappa


def compute_lagrange(points, samples=1000):
    """
    Calcule les points (x, y) et la courbure (K, t) de l'interpolation de Lagrange.
    """
    n_points = len(points)
    T = np.linspace(0, n_points - 1, samples)
    xs, ys = [], []
    K, t_global = [], []
    
    P_t = get_point_function(points) # le fait de retourner une fonction va faciliter les appel de dériver

    for t in T:
        # 1. Calcul du point (x, y)
        pt = P_t(t)
        xs.append(pt[0])
        ys.append(pt[1])

        # 2. Calcul des dérivées et de la courbure
        d1 = calculate_first_derivative(P_t, t)
        d2 = calculate_second_derivative(P_t, t)
        
        kappa = calculate_curvature_from_derivatives(d1, d2)
        
        K.append(kappa)
        t_global.append(t)

    # Retourne les 4 composantes pour la compatibilité avec compute_spline
    return np.array(xs), np.array(ys), np.array(K), np.array(t_global)


def lagrange_interpolation():
    """ Fonction de démonstration pour le main block, similaire à l'original. """
    # ax1 doit être défini globalement ou passé en paramètre dans le contexte __main__
    global ax1 
    xp,yp = PolygonAcquisition(ax1,'ob','--b')
    Points = [np.array([xp[i], yp[i]]) for i in range(len(xp))]
    
    x, y, _, _ = compute_lagrange(Points)

    ax1.plot(x,y,'r')
    plt.draw()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    minmax = 7
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,10))
    ax1.set_xlim((-minmax,minmax))
    ax1.set_ylim((-minmax,minmax))
    ax1.set_xlabel('Axe des x')
    ax1.set_ylabel('Axe des y')
    ax1.grid(True)
    ax2.grid(True)
    
    # Sélection du Mode d'Acquisition
    mode = input("Mode d'acquisition pour Lagrange : (M)anuelle ou (G)raphique ? [G/M] ").strip().upper()
    
    xp, yp = [], []

    if mode == 'M':
        # Mode Saisie Manuelle
        xp, yp = ManualPointAcquisition()
        ax1.set_title("Tracé des points saisis manuellement")
        
    elif mode == 'G':
        # Mode Graphique (PolygonAcquisition)
        ax1.set_title("Acquisition graphique (Clic droit pour terminer)")
        xp, yp = PolygonAcquisition(ax1,'ob','--b')
        
    else:
        print("Mode non reconnu. Abandon.")
        plt.close(fig)
        exit()

    if len(xp) < 2:
        print("Il faut au moins deux points pour continuer.")
        plt.close(fig)
        exit()
    
    Points = [np.array([xp[i], yp[i]]) for i in range(len(xp))]


    xs, ys, K, t_global = compute_lagrange(Points)

    
    if mode != 'G':
        ax1.plot(xp, yp, '--b', label='Polygone de contrôle')
    
    ax1.plot(xp, yp, 'ko', label='Points de contrôle', ms=8)
    
    # Tracé de la courbe sur ax1
    ax1.plot(xs, ys, 'r', label='Lagrange Interpolation')
    ax1.set_title("Interpolation de Lagrange via Aitken–Neville")
    ax1.legend()
    
    # Tracé de la courbure sur ax2
    ax2.plot(t_global, K, 'r')
    ax2.set_xlabel('Paramètre t')
    ax2.set_ylabel('Courbure $\\kappa$')
    ax2.set_title('Courbure de l\'interpolation de Lagrange')
    
    plt.tight_layout()
    plt.show()