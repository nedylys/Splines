
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.special import binom

minmax = 7
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_xlim((-minmax,minmax))
ax.set_ylim((-minmax,minmax))
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_title("Acquisition window") 
ax.grid(True)

def bezier_cubic(t, p0, p1, p2, p3):
    """
    Cubic Bézier interpolation using 4 control points.

    This is a special-case optimization for Bézier curves of degree 3.
    It uses the expanded Bernstein polynomial form for faster computation.
    """

    return ((1 - t)**3 * p0 +
            3 * (1 - t)**2 * t * p1 +
            3 * (1 - t) * t**2 * p2 +
            t**3 * p3)

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


def ComputeTangents_equidistant(Points, c) :
    """ Compute the tangent vectors at all points
        for a spline with equidistant parameterization
        and tension parameter c
    """
    n_points = len(Points) - 1
    m = [np.zeros(2) for _ in range(n_points + 1)]
    m[0] = ComputeTangentVectors(Points[0], Points[1], 0, 1, c)
    for i in range(1,n_points-1):
        m[i] = ComputeTangentVectors(Points[i-1], Points[i+1], i-1, i+1, c)
    m[n_points] = ComputeTangentVectors(Points[n_points - 1], Points[n_points], n_points - 1, n_points, c )
    return m

def splinesHermite():
    c = float(input("Enter tension parameter c (0 for Catmull-Rom, 1 for linear): "))
    xp, yp = PolygonAcquisition(ax,'ob','--b')
    n_points = len(xp)
    if n_points < 2:
        print("Il faut au moins deux points.")
        return
    n_segments = n_points - 1
    Points = [np.array([xp[i], yp[i]]) for i in range(n_points)]
    m = ComputeTangents_equidistant(Points, c)
    #print("Number of points acquired:", n_points)

    Bezier_segments = []
    t_segment = np.linspace(0, 1, 100)

    for i in range(n_segments):
        Bezier_segment = []

        B0, B1, B2, B3 = Hermite2Bezier(Points[i], Points[i+1], m[i], m[i+1])
        Bezier_segment = np.array([bezier_cubic(t, B0, B1, B2, B3) for t in t_segment])
        
        #Bezier_segments.append(Bezier_segment)
        ax.plot(Bezier_segment[:,0], Bezier_segment[:,1], 'r')
        plt.draw()
        
        #Bezier_segments.append(Bezier_segment)


    plt.title("Hermite Spline Interpolation")
    plt.axis('equal')
    plt.show()

splinesHermite()