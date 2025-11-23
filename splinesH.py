
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
    """ Acquisition of a 2D polygon.
        Left-click to add points. Close the figure window to stop acquisition.
    """
    x = []
    y = []
    
    # Use ginput with no mouse buttons specified, stop by closing the window
    # The first element of the list `coord` is the list of (x,y) tuples
    coord = plt.ginput(-1, mouse_add=1, mouse_pop=2) # -1 means infinite clicks until the figure is closed or ESC is pressed

    if coord:
        for xx, yy in coord:
            x.append(xx)
            y.append(yy)
            ax.plot(xx, yy, color1, ms=8)
        
        # Plot the segments after all points are acquired
        if len(x) > 1:
            ax.plot(x, y, color2)
            
    plt.draw() # Force a final draw
    return x, y

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

def compute_spline(points, order, tension=None):
    n_points = len(points)
    n_segments = n_points - 1
    
    if order == 1:
        if tension is None:
            raise ValueError("C1 spline requires tension parameter.")
        m = ComputeTangents_equidistant(points, tension)
    elif order == 2:
        m = ComputeTangentVectors_C2(points)

    xs = []
    ys = []
    K = []
    t_global = []

    t_segment = np.linspace(0, 1, 500)

    for i in range(n_segments):
        B0, B1, B2, B3 = Hermite2Bezier(points[i], points[i+1], m[i], m[i+1])
        segment = np.array([bezier_cubic(t, B0, B1, B2, B3) for t in t_segment])
        
        xs.extend(segment[:,0])
        ys.extend(segment[:,1])

        calculCourbure(i, B0, B1, B2, B3, K, t_global)

    return np.array(xs), np.array(ys), np.array(K), np.array(t_global)



def ManualPointAcquisition():
    """ 
    Acquisition manuelle des points de contrôle via le terminal.
    L'utilisateur saisit les coordonnées x et y sous forme de listes séparées par des virgules.
    """
    print("\n--- Saisie Manuelle des Points de Contrôle ---")
    while True:
        try:
            # Saisie des x
            xp_str = input("Entrez les coordonnées X (séparées par des virgules, ex: 1.0, 2.5, 3): ")
            xp = [float(x.strip()) for x in xp_str.split(',')]
            
            # Saisie des y
            yp_str = input("Entrez les coordonnées Y (séparées par des virgules, ex: 0.0, 1.0, 0.5): ")
            yp = [float(y.strip()) for y in yp_str.split(',')]
            
            if len(xp) != len(yp):
                print("Erreur: Le nombre de coordonnées X et Y doit être identique.")
                continue
            if len(xp) < 2:
                print("Erreur: Il faut au moins deux points.")
                continue
                
            return xp, yp
        except ValueError:
            print("Erreur: Veuillez n'entrer que des nombres séparés par des virgules.")
            


if __name__ == "__main__":
    minmax = 7
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
    ax1.set_xlim((-minmax,minmax))
    ax1.set_ylim((-minmax,minmax))
    ax1.set_xlabel('Axe des x')
    ax1.set_ylabel('Axe des y')
    ax1.grid(True)
    ax2.grid(True)
    
    
    # Demande de l'Ordre
    Ordre = int(input("Entrer l'ordre de la spline (1 pour C1, 2 pour C2): "))

    # Demande de la Tension (si C1)
    tension = None
    if Ordre == 1:
        tension = float(input("Entrer le parametre de tension c (0 pour Catmull-Rom, 1 pour linear): "))

    # Demande du Mode d'Acquisition
    mode = input("Mode d'acquisition : (M)anuelle ou (G)raphique ? [G/M] ").strip().upper()
    
    xp, yp = [], []
        # Variables fournies pour le test du contre-exemple de convex
    """ xp = [0, 4.5, 5, 4.5, 2.5, 0, 0]
    yp = [0, 0, 1.5, 3, 5, 3, 0] """
    # Monotonie
    """ xp = [-5.0, -3.0, -2.0, 0.0, 6.0]
    yp = [1.0, 1.1, 5.0, 5.5, 6.0] """
    # Ondulations
    """ xp = [-6.0, -3.0, -2.5, 0.0, 6.0]
    yp = [0.0, 0.0, 6.0, 0.0, 0.0]
 """
    # Coeur
    """ xp = [-3, -4, -3, -1, 0, 1, 3, 4, 3, 2, 1, 0, -1, -2, -3]
    yp = [ 2,  4,  6,  7, 6, 7, 6, 4, 2, 0, -2, -4, -2,  0,  2] """

    # Baguette
    """ xp = [-7, -6, -4, 0, 4, 6, 7, 6, 4, 0, -4, -6, -7]
    yp = [ 0,  2,  3, 3, 3, 2, 0, -2, -3, -3, -3, -2,  0] """

    # Voiture
    """ xp = [-4.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 4.0, -4.0]
    yp = [-4.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0, -4.0, -4.0]
 """

    if mode == 'M':
        # Saisie Manuelle
        ax1.set_title("Saisie Manuelle (Voir Terminal)")
        xp, yp = ManualPointAcquisition()
        
    elif mode == 'G':
        # Acquisition Graphique (PolygonAcquisition)
        # La fenêtre s'ouvre. Le terminal attend ici.
        ax1.set_title("Acquisition Graphique (Clic droit ou Entrée pour terminer)")
        xp, yp = PolygonAcquisition(ax1,'ob','--b')
        
    else:
        print("Mode non reconnu. Abandon.")
        plt.close(fig)
        exit()
    if len(xp) < 2:
        print("Il faut au moins deux points pour continuer.")
        plt.close(fig)
        exit()

    n_points = len(xp)
    Points = [np.array([xp[i], yp[i]]) for i in range(n_points)]

    # Tracé des points et du polygone
    if mode == 'M':

        ax1.plot(xp, yp, '--b', label='Polygone de contrôle')
    ax1.plot(xp, yp, 'ko', label='Points de contrôle', ms=8)
    

    xs, ys, K, t_global = compute_spline(Points, Ordre, tension=tension)

    ax1.plot(xs, ys, 'r', label=f'Spline C{Ordre}')
    ax1.set_title(f"Interpolation de la Spline C{Ordre}")
    ax1.legend()
    
    ax2.plot(t_global, K, 'r')
    ax2.set_xlabel('Paramètre t')
    ax2.set_ylabel('Courbure')
    ax2.set_title(f'Courbure de la Spline C{Ordre}')
    
    plt.tight_layout()
    plt.show()

    
