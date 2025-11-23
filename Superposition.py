import matplotlib.pyplot as plt
import numpy as np

# Assurez-vous d'utiliser le nom de fichier correct pour l'importation
from Lagrange import compute_lagrange 
from splinesH import compute_spline, PolygonAcquisition, ManualPointAcquisition


def main():
    """
    Fonction principale du programme. Elle gère l'acquisition des points de contrôle,
    calcule les interpolations (C1 Spline, C2 Spline, Lagrange) et ouvre 2 fenêtres 
    pour visualiser les résultats :
    
    1. Figure 1 (2x1 subplots) :
       - Courbes d'interpolation superposées.
       - Courbures des trois méthodes superposées.
       
    2. Figure 2 (2x3 subplots) :
       - Analyse détaillée et séparée de chaque méthode (C1, C2, Lagrange).
       - Ligne supérieure : Courbe d'interpolation (à l'échelle complète).
       - Ligne inférieure : Courbe de courbure correspondante ( vs. t).
    
    L'utilisateur est invité à saisir la tension C1 et les points de contrôle via l'interface graphique.
    """
    # Définition des limites initiales de la fenêtre d'acquisition et de superposition
    minmax = 7
    # 1. Figure Principale: Courbes Superposées (ax1) et Courbures Superposées (ax2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Configuration des axes pour l'acquisition et la superposition (ax1)
    ax1.set_xlim((-minmax, minmax))
    ax1.set_ylim((-minmax, minmax))
    ax1.set_xlabel('Axe des x')
    ax1.set_ylabel('Axe des y')
    ax1.grid(True)
    ax2.grid(True)
    
    # Demande du paramètre de tension pour la spline C1 
    tension = float(input("Entrez la tension pour la spline C1 (0=Catmull-Rom, 1=linéaire): "))

    # Choix du Mode d'Acquisition
    mode = input("Mode d'acquisition : (M)anuelle ou (G)raphique ? [G/M] ").strip().upper()
    xp, yp = [], []

    # Acquisition des points ---
    if mode == 'M':
        xp, yp = ManualPointAcquisition()
        ax1.set_title("Tracé des points saisis manuellement")
        
    elif mode == 'G':
        ax1.set_title("Cliquer pour placer les points (La fenêtre ou appuyer sur Entrée pour terminer)")
        xp, yp = PolygonAcquisition(ax1, 'ob', '--b')
        
    else:
        print("Mode non reconnu. Abandon.")
        plt.close(fig)
        return

    if len(xp) < 2:
        print("Il faut au moins 2 points.")
        plt.close(fig) 
        return
    points = np.column_stack((xp, yp))
    
    if mode == 'M':
        ax1.plot(xp, yp, 'ko', label='Points de contrôle', zorder=5, ms=8)
        ax1.plot(xp, yp, '--k', label='Polygone de contrôle')
    elif mode == 'G':
        ax1.plot(xp, yp, 'ko', label='Points de contrôle', zorder=5, ms=8)
        ax1.plot(xp, yp, '--k', label='Polygone de contrôle')
    
    # Calcul des courbes et courbures 
    try:
        xs_c1, ys_c1, K_c1, t_c1 = compute_spline(points, order=1, tension=tension)
        xs_c2, ys_c2, K_c2, t_c2 = compute_spline(points, order=2)
        xs_lg, ys_lg, K_lg, t_lg = compute_lagrange(points) # Récupération de K_lg et t_lg
    except ValueError as e:
        print(f"Erreur de calcul: {e}")
        plt.close(fig)
        return


    #####################################################################
    #######       1. Tracé de la Figure de Superposition          #######
    #####################################################################
    
    # Tracé des courbes sur l'axe de superposition (ax1)
    ax1.plot(xs_c1, ys_c1, 'r', label='Spline C1 (Hermite)')
    ax1.plot(xs_c2, ys_c2, 'g', label='Spline C2')
    ax1.plot(xs_lg, ys_lg, 'b', label='Interpolation de Lagrange')
    
    # Fixer les limites
    ax1.set_xlim((-minmax, minmax))
    ax1.set_ylim((-minmax, minmax)) 
    ax1.set_title("Superposition des courbes")
    ax1.axis('equal') # Assure des échelles égales en x et y
    ax1.grid(True)
    # Ajouter la légende
    ax1.legend()

    # Fixer à nouveau les limites pour garantir que le zoom ne s'adapte pas à Lagrange
    ax1.set_xlim((-minmax, minmax))
    ax1.set_ylim((-minmax, minmax))
    

    
    # Tracé des Courbures Superposées sur l'axe (ax2)
    ax2.plot(t_c1, K_c1, 'r', label='Courbure C1')
    ax2.plot(t_c2, K_c2, 'g', label='Courbure C2')
    ax2.plot(t_lg, K_lg, 'b--', label='Courbure Lagrange') 
    ax2.set_xlabel('Paramètre t (segment)')
    ax2.set_ylabel('Courbure')
    ax2.set_title("Courbure des Splines et de Lagrange (Superposées)")
    ax2.grid(True)
    ax2.legend()
    
    ####################################################################
    ## 2. Nouvelle Figure: Tracés Séparés (Interpolation + Courbure)  ##
    ####################################################################
    
    # Crée une figure avec 2 lignes et 3 colonnes pour l'analyse séparée
    fig_separate, ((ax3, ax4, ax5), (ax6, ax7, ax8)) = plt.subplots(2, 3, figsize=(18, 12))
    fig_separate.suptitle('Analyse Séparée des Courbes et Courbures', fontsize=16)
    
    control_points_scatter = {'x': points[:,0], 'y': points[:,1], 'c': 'k', 'marker': 'o', 'label': 'Points de contrôle'}
    
    # Récupérer la plage max pour uniformiser l'échelle des courbes (y compris Lagrange)
    all_x = np.concatenate([xs_c1, xs_c2, xs_lg])
    all_y = np.concatenate([ys_c1, ys_c2, ys_lg])
    max_range = max(np.max(all_x), np.max(all_y))
    min_range = min(np.min(all_x), np.min(all_y))
    padding = 1.0 # Marge autour des limites, pour que les courbes ne touchent pas aux bords des graphiques

    # Colonne 1: C1 Spline
    # Ligne 1: Courbe (ax3)
    ax3.plot(xs_c1, ys_c1, 'r', label='Spline C1')
    ax3.grid(True)
    ax3.scatter(**control_points_scatter, zorder=5)
    ax3.set_title('Interpolation Spline C1')
    # Application des limites pour voir la courbe entière (y compris les pics de Lagrange si elle est large)
    ax3.set_xlim(min_range - padding, max_range + padding)
    ax3.set_ylim(min_range - padding, max_range + padding)
    ax3.axis('equal')
    ax3.legend()
    
    # Ligne 2: Courbure (ax6)
    ax6.plot(t_c1, K_c1, 'r', label='Courbure C1')
    ax6.grid(True)
    ax6.set_title('Courbure C1')
    ax6.set_xlabel('t')
    ax6.legend()

    # Colonne 2: C2 Spline
    # Ligne 1: Courbe (ax4)
    ax4.plot(xs_c2, ys_c2, 'g', label='Spline C2')
    ax4.grid(True)
    ax4.scatter(**control_points_scatter, zorder=5)
    ax4.set_title('Interpolation Spline C2')
    ax4.set_xlim(min_range - padding, max_range + padding)
    ax4.set_ylim(min_range - padding, max_range + padding)
    ax4.axis('equal')
    ax4.legend()
    
    # Ligne 2: Courbure (ax7)
    ax7.plot(t_c2, K_c2, 'g', label='Courbure C2')
    ax7.grid(True)
    ax7.set_title('Courbure C2')
    ax7.set_xlabel('t')
    ax7.legend()
    
    # Colonne 3: Lagrange Interpolation 
    # Ligne 1: Courbe (ax5)
    ax5.plot(xs_lg, ys_lg, 'b', label='Lagrange')
    ax5.grid(True)
    ax5.scatter(**control_points_scatter, zorder=5)
    ax5.set_title('Interpolation Lagrange (Plage Complète)')
    ax5.set_xlim(min_range - padding, max_range + padding)
    ax5.set_ylim(min_range - padding, max_range + padding)
    ax5.axis('equal')
    ax5.legend()

    # Ligne 2: Courbure (ax8)
    ax8.plot(t_lg, K_lg, 'b', label='Courbure Lagrange')
    ax8.grid(True)
    ax8.set_title('Courbure Lagrange')
    ax8.set_xlabel('t')
    ax8.legend()
    
    # Ajouter des labels génériques pour l'ensemble des axes
    for ax in [ax3, ax4, ax5]:
         ax.set_xlabel('Axe des x')
         ax.set_ylabel('Axe des y')
    for ax in [ax6, ax7, ax8]:
         ax.set_ylabel('Courbure')
         

    plt.show()

if __name__ == "__main__":
    main()