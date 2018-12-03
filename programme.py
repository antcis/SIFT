# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import numpy as np
import math
import cv2
import fonctions as fc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import random


plt.close('all')

#Afin d'observer nos différents tests, décommenter les différentes sous-parties en fin de code.


#Affichage du lieu des Points-Clés à partir de la liste des Points-Clés orientés
def affichagePointsCles(im, listeAvecOrientation) :
    print('...Affichage des résultats')
    fig, axd = plt.subplots(1)
    axd.imshow(im, cmap='gray')
    for liste in listeAvecOrientation :
        [o,s,x,y,sigma,deltaO,theta] = liste
        circ = patches.Wedge((int(y),int(x)),sigma, 0, 360, width=1, color='red')
        axd.add_patch(circ)
        axd.add_line(mlines.Line2D([int(y), int(y+sigma*math.sin(theta))],[int(x),int(x+sigma*math.cos(theta))]))
    plt.title('Affichage des lieux des points-clés')
    plt.show()
    
#Affichage du matching de deux images à partir de la liste des descripteurs
def matching(img, imd, KFgauche, KFdroite, n, r) :
    print('...Affichage des résultats')
    fig,(axg,axd) = plt.subplots(1,2)    
    axg.imshow(img, cmap='gray')
    axd.imshow(imd, cmap='gray')    
    mat = fc.matriceDeDistance(KFgauche,KFdroite)
    mins = fc.trouverMins(mat, n)    
    for i in mins:
        gauche = i[0]
        droite = i[1]        
        color = '#{}'.format("%06x" % random.randint(0, 0xFFFFFF))    
        circg = patches.Wedge((int(KFgauche[gauche,1]),int(KFgauche[gauche,0])),r, 0, 360, width=1, color=color)
        axg.add_patch(circg)
        circd = patches.Wedge((int(KFdroite[droite,1]),int(KFdroite[droite,0])),r, 0, 360, width=1, color=color)
        axd.add_patch(circd)        
    plt.title('Matching des points clés')
    plt.show()
    
##Fonction trouvant les points clés d'une image avec affichage des résultats
    #Nspo = nombre d'échelles par octave
    #Noct = nombre d'octaves
def trouverPointsCles(nomImage, Nspo, Noct) :
    print('### Image '+nomImage)
    img = cv2.imread(nomImage, 0)
    imGauche = img/255
    print('...Pyramide de Gauss')
    pyramGauss = fc.genOctavesGauss(imGauche, Nspo, Noct)
    print('...Différence de Gaussiennes')
    diffGauss = fc.diffGaussiennes(imGauche, Nspo, Noct)
    print('...Détection de points clés')
    pointsCles, listeCompteur, listeResolution = fc.pointsClesBasique(diffGauss, pyramGauss, Nspo, Noct)
    print('...Calcul du gradient')
    grad_ligne, grad_col = fc.computeGradients2(pyramGauss, Noct, Nspo)
    print('...Détermination des orientations')
    listeAvecOrientation = fc.computeOrientation2(pointsCles, grad_ligne, grad_col)
    matrice_pointscles = np.array([[i[2], i[3], i[4], i[6]] for i in listeAvecOrientation])
    np.save('mat.npy', matrice_pointscles)
    affichagePointsCles(img, listeAvecOrientation)
    
##Fonction trouvant les points clés d'une image avec affichage des résultats
    #Nspo = nombre d'échelles par octave
    #Noct = nombre d'octaves
    #n = nombre de points pour effectuer le matching
    #r: valeur modifiant au rayon affiché sur l'image, 10 convient pour des images petites, 30 pour des dimensions égales à 1000
def trouverDescripteurs(nomImageGauche, nomImageDroite, Nspo, Noct, n=10, r=30):
    ### GAUCHE
    
    print('### Image '+nomImageGauche)
    img = cv2.imread(nomImageGauche, 0)
    imGauche = img/255
    print('...Pyramide de Gauss')
    pyramGauss = fc.genOctavesGauss(imGauche, Nspo, Noct)
    print('...Différence de Gaussiennes')
    diffGauss = fc.diffGaussiennes(imGauche, Nspo, Noct)
    print('...Détection de points clés')
    pointsCles, listeCompteur, listeResolution = fc.pointsClesBasique(diffGauss, pyramGauss, Nspo, Noct)
    print('...Calcul du gradient')
    grad_ligne, grad_col = fc.computeGradients2(pyramGauss, Noct, Nspo)
    print('...Détermination des orientations')
    listeAvecOrientation = fc.computeOrientation2(pointsCles, grad_ligne, grad_col)
    matrice_pointscles = np.array([[i[2], i[3], i[4], i[6]] for i in listeAvecOrientation])
    np.save('matGauche.npy', matrice_pointscles)
    print('...Calcul des descripteurs')
    KPgauche = fc.keypoint_descriptor(listeAvecOrientation, grad_ligne, grad_col)

    ### DROITE

    print('### Image '+nomImageDroite)
    imd = cv2.imread(nomImageDroite, 0)
    imDroite = imd/255
    print('...Pyramide de Gauss')
    pyramGauss = fc.genOctavesGauss(imDroite, Nspo, Noct)
    print('...Différence de Gaussiennes')
    diffGauss = fc.diffGaussiennes(imDroite, Nspo, Noct)
    print('...Détection de points clés')
    pointsCles, listeCompteur, listeResolution = fc.pointsClesBasique(diffGauss, pyramGauss, Nspo, Noct)
    print('...Calcul du gradient')
    grad_ligne, grad_col = fc.computeGradients2(pyramGauss, Noct, Nspo)
    print('...Détermination des orientations')
    listeAvecOrientation = fc.computeOrientation2(pointsCles, grad_ligne, grad_col)
    matrice_pointscles = np.array([[i[2], i[3], i[4], i[6]] for i in listeAvecOrientation])
    np.save('matDroite.npy', matrice_pointscles)
    print('...Calcul des descripteurs')
    KPdroite = fc.keypoint_descriptor(listeAvecOrientation, grad_ligne, grad_col)

    ### MATCHING
    print('...Affichage des résultats')
    matching(img, imd, KPgauche, KPdroite, n, r)
 
    
    
## But : Observer les lieux des points-clés de Mars avec orientation
#trouverPointsCles('gauche.jpg', 3, 4) 
    
## But : Observer les lieux des points-clés de l'image de Lenna (512*512) et l'image réduite de moitié (256*256)
#trouverPointsCles('Lenna.jpg', 3, 4)
#trouverPointsCles('Lenna_Small.jpg', 3, 4)
        
## But : Observer les lieux des points-clés de l'image de Lenna et l'image de Lenna tournée de 90 ° (255*255)
#trouverPointsCles('Lenna_Small.jpg', 3, 4)
#trouverPointsCles('Lenna_SmallRotation.jpg', 3, 4)
    

## But : Observer Le Matching entre les images Lenna Gauche et Lenna Droite
#trouverDescripteurs('LennaGauche.jpg', 'LennaDroite.jpg', 3, 4)

## But : Observer Le Matching entre les deux images de Mars (très long en temps de calcul des descripteurs)
#trouverDescripteurs('gauche.jpg', 'droite.jpg', 3, 4)