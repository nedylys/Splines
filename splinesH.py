
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import binom




def bezier_cubic(t, p0, p1, p2, p3):
    """
    Cubic BÃ©zier interpolation using 4 control points.

    This is a special-case optimization for BÃ©zier curves of degree 3.
    It uses the expanded Bernstein polynomial form for faster computation.
    """

    return ((1 - t)**3 * p0 +
            3 * (1 - t)**2 * t * p1 +
            3 * (1 - t) * t**2 * p2 +
            t**3 * p3)

def bezier_cubic_prime(t, p0, p1, p2, p3):
    """ 
    First derivative of a cubic BÃ©zier curve.
    """
    return (3 * (1 - t)**2 * (p1 - p0) +
            6 * (1 - t) * t * (p2 - p1) +
            3 * t**2 * (p3 - p2))

def bezier_cubic_second(t, p0, p1, p2, p3):
    """
    Second derivativeof a cubic BÃ©zier curve.
    """
    return (6 * (1 - t) * (p2 - 2 * p1 + p0) +
            6 * t * (p3 - 2 * p2 + p1))

def PolygonAcquisition(ax,color1,color2) :
    """ Acquisition of a 2D polygon in the window with subplot "ax" 
        right click stop the acquisition
    """
    x = [] # x is an empty list
    y = []
    coord = 0
    while coord != []:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        # coord is a list of tuples : coord = [(x,y)]
        if coord != []:
            xx = coord[0][0]
            yy = coord[0][1]
            ax.plot(xx,yy,color1,ms=8)
            x.append(xx)
            y.append(yy)
            plt.draw()
            if len(x) > 1 :
                ax.plot([x[-2],x[-1]],[y[-2],y[-1]],color2)
    return x,y

def Hermite2Bezier(P0,P1,T0,T1) :
    """ Conversion of a Hermite spline defined by points P0 and P1
        and tangents T0 and T1 into a Bezier spline defined by
        control points B0, B1, B2, B3
    """
    B0 = P0
    B1 = P0 + T0/3.0
    B2 = P1 - T1/3.0
    B3 = P1
    return B0,B1,B2,B3

def ComputeTangentVectors(P1, P2, u1, u2, c) :
    """ Compute the tangent vectors at points P1 and P2
        for a spline with chord length parameterization
        and tension parameter c
    """
    m_k = (1 - c)*((P2 - P1)/ (u2 - u1))
    return m_k


def ComputeTangents_equidistant(Points, c):
    """Compute tangents for equidistant parameterization (Catmull-Rom like).
       Points: list of 2D numpy arrays.
    """
    n_points = len(Points)
    m = [np.zeros(2) for _ in range(n_points)]
    if n_points == 1:
        return m
    # endpoints
    m[0] = ComputeTangentVectors(Points[0], Points[1], 0, 1, c)
    for i in range(1, n_points - 1):
        m[i] = ComputeTangentVectors(Points[i-1], Points[i+1], i-1, i+1, c)
    m[-1] = ComputeTangentVectors(Points[-2], Points[-1], n_points - 2, n_points - 1, c)
    return m



def calculCourbure(i_segment, B0, B1, B2, B3, K, T):
    """Calcul de la courbure Îº(t) pour le segment i_segment."""
    t_intervalle = np.linspace(0, 1, 100)
    for t in t_intervalle:
        T.append(t + i_segment)
        d1 = bezier_cubic_prime(t, B0, B1, B2, B3)
        d2 = bezier_cubic_second(t, B0, B1, B2, B3)
        # Îº = |x'y'' - y'x''| / ( (x'^2 + y'^2)^(3/2) )
        numerateur = d1[0]*d2[1] - d1[1]*d2[0]
        denominateur = np.linalg.norm(d1)**3
        kappa = numerateur / denominateur if denominateur > 1e-10 else 0
        K.append(kappa)


def Cholesky_Factorization(Diag, Diagsym):
    """ 
    In our case, we have a symmetric tridiagonal matrix with a strictly positive diagonal. 
    It is defined by its diagonal Diag and its sub-diagonal Diagsym, so there is no need to store the entire matrix (only zeros that would waste memory).
    The function returns the Cholesky factorization in the form of a list LDiag (the diagonal of L) and LDiagsym (the sub-diagonal of L).
    """
    n = len(Diag)
    if n < 2:
        raise ValueError("The spline must have at least two points.")
    T_Diag = np.zeros(n) # diagonal of T (Matrice de Cholesky)
    T_ss_Diag = np.zeros(n-1) # sub-diagonal of T
    T_Diag[0] = np.sqrt(Diag[0])
    for i in range(1,n):
        T_ss_Diag[i-1] = Diagsym[i-1]/T_Diag[i-1]
        under_sqrt = Diag[i] - T_ss_Diag[i-1]**2
        if under_sqrt <= 0:
            raise ValueError("Matrix is not positive definite; Cholesky factorization failed.")
        T_Diag[i] = np.sqrt(under_sqrt)
    return T_Diag, T_ss_Diag


def Solve_Cholesky(T_Diag, T_ss_Diag, B):
    """ 
        Solving the system T.transp(T).X=BT
        where T is a lower triangular matrix
        defined by its diagonal T_Diag and its sub-diagonal T_ss_Diag.
    """
    B = np.asarray(B, dtype=float) # ensure B is a numpy array of floats, if it is already an array it will not be changed (asarray makes no copies in that case)
    n = len(B)
    Y = np.zeros(n)
    # Resolution of TY = B, where Y = transp(T).X with an ascending recurrence
    Y[0] = B[0]/T_Diag[0]
    for i in range(1,n):
        Y[i] = (B[i] - T_ss_Diag[i-1]*Y[i-1])/T_Diag[i]
    # Resolution of transp(T).X = Y with a descending recurrence
    X = np.zeros(n)
    X[n-1] = Y[n-1]/T_Diag[n-1]
    for i in range(n-2,-1,-1):
        X[i] = (Y[i] - T_ss_Diag[i]*X[i+1])/T_Diag[i]
    return X


def ComputeTangentVectors_C2(Points):
    """Tangentes pour spline cubique C2 interpolante, stable pour n >= 2"""
    n = len(Points)
    if n == 2:  # cas particulier 2 points
        d = Points[1] - Points[0]
        return [d.copy(), d.copy()]
    
    Diag = np.zeros(n)
    Diagsym = np.ones(n-1)  # sous-diagonale = 1 partout
    Bx = np.zeros(n)
    By = np.zeros(n)

    # coins
    Diag[0] = 2.0
    Diag[-1] = 2.0
    Bx[0] = 3.0 * (Points[1][0] - Points[0][0])
    By[0] = 3.0 * (Points[1][1] - Points[0][1])
    Bx[-1] = 3.0 * (Points[-1][0] - Points[-2][0])
    By[-1] = 3.0 * (Points[-1][1] - Points[-2][1])

    # points intÃ©rieurs
    for i in range(1, n-1):
        Diag[i] = 4.0
        Bx[i] = 3.0 * (Points[i+1][0] - Points[i-1][0])
        By[i] = 3.0 * (Points[i+1][1] - Points[i-1][1])

    # factorisation + rÃ©solution
    T_Diag, T_ss_Diag = Cholesky_Factorization(Diag, Diagsym)
    mx = Solve_Cholesky(T_Diag, T_ss_Diag, Bx)
    my = Solve_Cholesky(T_Diag, T_ss_Diag, By)
    
    m = [np.array([mx[i], my[i]]) for i in range(n)]
    return m




def splines(Order):
    if Order == 1:
        c = float(input("Enter tension parameter c (0 for Catmull-Rom, 1 for linear): "))
    xp, yp = PolygonAcquisition(ax1,'ob','--b')
    #xp, yp = [0, -1.5, -2.5, -2.75, -2, -0.75, 0, 0.75, 2, 2.75, 2.5, 1.5, 0],[-6, -3, -1, 2, 4, 5, 3, 5, 4, 2, -1, -3, -6]
    n_points = len(xp)
    if n_points < 2:
        print("Il faut au moins deux points.")
        return
    n_segments = n_points - 1
    Points = [np.array([xp[i], yp[i]]) for i in range(n_points)]

    if Order == 1:
        m = ComputeTangents_equidistant(Points, c)
    elif Order == 2:
        m = ComputeTangentVectors_C2(Points)

    #print("Number of points acquired:", n_points)

    t_segment = np.linspace(0, 1, 500)
    K = []
    t_global = []
    ax1.set_title("Spline C2")
    for i in range(n_segments):
        Bezier_segment = []

        B0, B1, B2, B3 = Hermite2Bezier(Points[i], Points[i+1], m[i], m[i+1])
        
        Bezier_segment = np.array([bezier_cubic(t, B0, B1, B2, B3) for t in t_segment])
        
        calculCourbure(i,B0,B1,B2,B3,K,t_global)
        #Bezier_segments.append(Bezier_segment)
        ax1.plot(Bezier_segment[:,0], Bezier_segment[:,1], 'r')
        plt.draw()
    
    ax2.plot(t_global,K)
    ax2.set_xlabel('Parametre t')
    ax2.set_ylabel('Curvature Îº')
    ax2.set_title('Curvature of the spline')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__" :
        minmax = 7
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,10))
        ax1.set_xlim((-minmax,minmax))
        ax1.set_ylim((-minmax,minmax))
        ax1.set_xlabel('x-axis')
        ax1.set_ylabel('y-axis')
        ax1.set_title("Acquisition window") 
        ax1.grid(True)
        Ordre = int(input("Enter spline order (1 for C1 (Hermite splines), 2 for C2): "))
        splines(Ordre)