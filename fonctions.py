# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:30:30 2018

@author: anfrac
"""

import numpy as np
import math
import cv2
import scipy.misc as sc

def noyauGaussien(sigma) :
    ksize = np.ceil(6*sigma)
    if(ksize%2==0) :
        ksize = ksize+1
    x = np.arange(0, ksize, 1, float)
    y = x[:,np.newaxis]
    x0 = y0 = ksize // 2
    mat = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)
    return mat/sum(mat.flatten())

def flouGaussien(im, sigma) :
    noyau = noyauGaussien(sigma)
    imFlou = cv2.filter2D(im, ddepth=-1, kernel=noyau)
    return imFlou

def genOctavesGauss(imageIni, Nspo, Noct) :
    k = pow(2, 1/Nspo)
    imOct1 = []
    v10 = sc.imresize(imageIni, 200, interp='bilinear')/255
    v10 = flouGaussien(v10, 1/0.5*math.sqrt(0.8*0.8-0.5*0.5)) #point 6 article sujet
    imOct1.append(v10)
    fact = 0.8/0.5 #point 7 article sujet
    for s in range(1, Nspo+3) :
        prochain = flouGaussien(imOct1[-1], fact*math.sqrt(pow(k,2*s)-pow(k,2*s-2)))
        imOct1.append(prochain)
    
    imTot = [imOct1]
    
    for o in range(2, Noct+1) :
        vo0 = sc.imresize(imTot[-1][-3], 50, interp='bilinear')/255
        imOctPro = [vo0]
        for s in range(1, Nspo + 3) :
            prochain = flouGaussien(imOctPro[-1], fact*math.sqrt(pow(k,2*s)-pow(k,2*s-2)))
            imOctPro.append(prochain)
        imTot.append(imOctPro)
        
    return imTot  
 
def diffGaussiennes(imageIni, Nspo, Noct) :
    imTot = genOctavesGauss(imageIni, Nspo, Noct)
    imDiffTot = []
    for i in range(1, Noct+1) :
        diffPro = []
        for s in range(0, Nspo+2):
            #imDiff = imTot[i-1][s+1].astype(np.int32)-imTot[i-1][s].astype(np.int32)
            imDiff = imTot[i-1][s+1]-imTot[i-1][s]
            diffPro.append(imDiff)
        imDiffTot.append(diffPro)
    return imDiffTot


def checkExtremas3(indOct, s, m, n, DoG) :
    elmt = DoG[indOct][s][m,n]
    liste = []
    for scale in range(s-1, s+2) :
        liste.append(DoG[indOct][scale][m-1:m+2,n-1:n+2])
    liste = np.array(liste)
    if((elmt==np.max(liste.flatten())) ^ (elmt==np.min(liste.flatten()))) :
        return True
    return False

    
def notdiscardContrast(indOct, s, m, n, DoG, seuil_contraste) :
    element = DoG[indOct][s][m,n]
    #np.abs?
    if(abs(element)>= seuil_contraste):
        return True
    return False

def notdiscardEdges(indOct, s, m, n, DoG, r_courb_principale) :
    hess = np.zeros((2,2))
    hess[0,0] = DoG[indOct][s][m + 1,n] + DoG[indOct][s][m - 1,n] -2 * DoG[indOct][s][m,n]
    hess[0,1] = 0.25 * (DoG[indOct][s][m+1, n+1] - DoG[indOct][s][m+1, n-1] - DoG[indOct][s][m-1, n+1] + DoG[indOct][s][m-1, n-1])
    hess[1,0] = hess[0,1]
    hess[1,1] = DoG[indOct][s][m, n+1] + DoG[indOct][s][m, n-1] -2 * DoG[indOct][s][m,n]
    edgeness = pow(np.trace(hess),2)/np.linalg.det(hess)
    ratio = pow(r_courb_principale+1, 2)/r_courb_principale
    if(edgeness < ratio) :
        return True
    return False

def pointsClesBasique(DoG, pyramide, Nspo, Noct) :
    delta_min = 0.5
    sigma_min = 0.8
    seuil_contraste = 0.03
    #THRESHOLD DE 10
    r_courb_principale = 10
    listePC = []
    #tableau permettant de gerer les pointscles
    liste_compteur = []
    liste_resolution = []
    for o in range(1, Noct+1) :
        #on renvoie la dog complete quid de la pyramide?
        octave = pyramide[o-1]
        resolution_octave = delta_min * pow(2, o-1)
        sigma = [(resolution_octave/delta_min)*sigma_min*pow(2, s/Nspo) for s in range(Nspo+3)] 
        listePCOctave, nombre_check = detectionPointsCles(o, DoG, octave, sigma, seuil_contraste, r_courb_principale, resolution_octave)
        #tableau permettant de gerer les pointscles 
        liste_compteur.append(nombre_check)
        liste_resolution.append(resolution_octave)
        listePC += listePCOctave
        #listePC = np.concatenate(listePC, listePCOctave)
    return listePC, liste_compteur, liste_resolution


#CEST PAS COURBPRINCIPALE CEST LE THRESHOLD
def detectionPointsCles(o, DoG, octave, sigma, seuil_contraste, r_courb_principale, resolution_octave) :
    #print('Octave '+str(o)+' en cours...')
    listePCOctave = []
    M,N = DoG[o-1][0].shape
    Nspo = len(octave) - 3 #douteux
    #la dog est entre 0 et nspo + 1 et la pyramide entre 0 et  nspo +2
    delta_min = 0.5
    comptCheck = 0
    comptContraste = 0
    comptExtremas = 0
    comptEdges = 0
    for s in range(1, Nspo+1) :
        print('   Scale '+ str(s) +' en cours...')
        for m in range(1, M-1) :
            for n in range(1, N-1) :
                comptCheck += 1
                if(notdiscardContrast(o-1, s, m, n, DoG, seuil_contraste)) :
                    comptContraste += 1
                    if(checkExtremas3(o-1, s, m, n, DoG)) :
                        comptExtremas += 1
                        if(notdiscardEdges(o-1, s, m, n, DoG, r_courb_principale)):
                            comptEdges += 1
                            #delta_min saute
                            deltaO = pow(2, o-1)*delta_min
                            sigma = deltaO/delta_min*0.8*pow(2,s/Nspo)
                            #print(sigma)
                            x = deltaO*m
                            y = deltaO*n
                            w = DoG[o-1][s][m,n]
                            listePCOctave.append([o,s,m,n,sigma,deltaO,x,y,w])
    print('        comptCheck='+str(comptCheck)+' ; comptContraste='+str(comptContraste)+' ; comptExtremas='+str(comptExtremas)+' ; comptEdges='+str(comptEdges))
    return listePCOctave, [comptCheck, comptContraste, comptExtremas, comptEdges]

def computeGradients2(pyramide, Noct, Nspo) :
    
    ligne = []
    col = []

    for o in range(1, Noct+1) :
        M,N = pyramide[o-1][0].shape
        ligneListe = []
        colListe = []
        for s in range(1, Nspo+1) :
            grad_ligne = np.zeros((M-2,N-2))
            grad_col = np.zeros((M-2,N-2)) 
            for i in range(1, M-1):
                for j in range(1, N-1):
                   grad_ligne[i-1,j-1] = (pyramide[o-1][s][i+1,j] - pyramide[o-1][s][i-1,j])/2
                   grad_col[i-1,j-1] = (pyramide[o-1][s][i,j+1] - pyramide[o-1][s][i,j-1])/2
            ligneListe.append(grad_ligne)
            colListe.append(grad_col)
        ligne.append(ligneListe) 
        col.append(colListe)
    return  ligne, col        
    
def computeOrientation2(listePC, grad_ligne, grad_col) :
    outListeOri = []
    lambdaOri = 1.5 #page 17
    #nOri = 8
    #nHist = 4
    nBins = 36
    t = 0.8
    for PC in listePC :
        [o,s,m,n,sigma,deltaO,x,y,w] = PC
        #s a toujours commence a 0, mais on nous a fait commencer le gradient a 1
        l,c = (len(grad_ligne[o-1][s-1]))*deltaO, (len(grad_ligne[o-1][s-1][0,:]))*deltaO
        fact = 3*lambdaOri*sigma
        #print('l='+str(l)+' ; c='+str(c)+' ; fact='+str(fact))
        #print('l ='+str(l)+' ; c='+str(c)+' ; l-fact='+str(l-fact)+' ; c-fact='+str(c-fact)+' ; fact='+str(fact))
        if((fact <= x) and (x < int(l-fact)) and (fact<=y) and (y< int(c-fact))) :
            histo = [0 for i in range(nBins)]
            for a in range(int(np.ceil((x-fact)/deltaO)), int(np.floor((x+fact)/deltaO)+1)) :
                for b in range(int(np.ceil((y-fact)/deltaO)), int(np.floor((y+fact)/deltaO)+1)) :
                    #print('a='+str(a)+' ; b='+str(b)+' ; m='+str(m)+' ; n='+str(n))
                    coriab = math.exp(-1*(np.linalg.norm([(a*deltaO-x), (b*deltaO-y)])**2/(2*(lambdaOri*sigma)**2)))*np.linalg.norm([grad_ligne[o-1][s-1][a-1,b-1], grad_col[o-1][s-1][a-1,b-1]])
                    #math.sqrt((grad_ligne[o-1][s-1][a-1,b-1]**2+grad_col[o-1][s-1][a-1,b-1]**2))
                    #print('coriab2')
                    #print(coriab)
                    boriab = int(nBins/(2*math.pi)*(np.arctan2(grad_col[o-1][s-1][a-1,b-1], grad_ligne[o-1][s-1][a-1,b-1]) % (2*math.pi)))
                    #print('boriab')
                    #print(boriab)
                    histo[boriab] += coriab
                    #print('   coriab='+str(coriab))
            kernel = np.array([1,1,1])/3
            #print('  histo before convolve=')
            #print(histo)
            #for i in range(6) :
            #    histo = np.convolve(histo,kernel)
           # print('  histo after convolve=') 
            #print(histo)
            for k in range(1, nBins+1) :
                k_indexed = k-1
                hk = histo[k_indexed]
                hkm = histo[(k_indexed-1)%nBins]
                hkp = histo[(k_indexed+1)%nBins]
                maxi = np.max(histo)
                #print(str(hk)+';hkm='+str(hkm)+';hkp='+str(hkp)+';tmaxi='+str(t*maxi))
                if((hk>hkm) and (hk>hkp) and (hk>t*maxi)) :
                    thetaK = 2*math.pi*(k-1)/nBins
                    theta = thetaK + 2*math.pi/nBins*(hkm-hkp)/(hkm-2*hk+hkp)
                    #outListe.append([o,s,m,n,deltaO,sigma,x,y,w,theta])
                    
            outListeOri.append([o,s,m*deltaO,n*deltaO,sigma,deltaO,theta])
    #print(histo)
    return outListeOri
                
def table(outListe):
    liste_adapt = []
    for PC in outListe : 
        [o,s,m,n,sigma, deltaO,x,y,w,theta]= PC
        lin = m*deltaO
        col = n*deltaO
        point = [o,s,col,lin,sigma,deltaO,theta] 
        liste_adapt.append(point)
    return liste_adapt   

def keypoint_descriptor(listePCori, grad_ligne, grad_col):
    outListeFeatured = []
    nhist = 4
    nOri = 8
    lambdaDescr = 6 #page 16
    #deviationWindow = lambdaDescr * sigmakey
    #Pdescr
    for PC in listePCori :
        [o,s,x,y,sigma, deltaO,theta] = PC
        feature = np.zeros(nhist*nhist*nOri)
        #s a toujours commence a 0, mais on nous a fait commencer le gradient a 1
        window = lambdaDescr * sigma
        patch = 2 * lambdaDescr * (nhist+1)/nhist * sigma
        l,c = (len(grad_ligne[o-1][s-1]))*deltaO, (len(grad_ligne[o-1][s-1][0,:]))*deltaO
        #SI PROBLEME OUT OF BOUND MULTIPLIER PAR N+1/N
        if((math.sqrt(2)*window*(nhist+1)/nhist <= x) and (x <= l-math.sqrt(2)*window*(nhist+1)/nhist) and ((math.sqrt(2)*window*(nhist+1)/nhist)<=y) and (y<=c-math.sqrt(2)*window*(nhist+1)/nhist)) :
            histo = np.zeros((nhist,nhist,nOri))
            for i in range(1,nhist+1):
                 for j in range(1,nhist+1):
                     for k in range(1,nOri+1):
                         histo[i-1,j-1,k-1] = 0
            for a in range(int(np.ceil((x - math.sqrt(2)*window*(nhist+1)/nhist)/deltaO)), int(np.floor((x + math.sqrt(2)*window*(nhist+1)/nhist)/deltaO))):
                for b in range(int(np.ceil((y - math.sqrt(2)*window*(nhist+1)/nhist)/deltaO)), int(np.floor((y + math.sqrt(2)*window*(nhist+1)/nhist)/deltaO))):
                    x_bis = ((a*deltaO - x)*math.cos(theta) +(b*deltaO - y)*math.sin(theta))/sigma
                    y_bis = (-(a*deltaO - x)*math.sin(theta) +(b*deltaO - y)*math.cos(theta))/sigma         
                    if (max(abs(x_bis),abs(y_bis)) < lambdaDescr*(nhist+1)/nhist):
                        theta_bis = (np.arctan2(grad_ligne[o-1][s-1][a-1,b-1],grad_col[o-1][s-1][a-1,b-1]) - theta)%(2*math.pi)
                        #QUIESTMODULO2PI
                        cdescr = math.exp(-1*(np.linalg.norm([(a*deltaO-x), (b*deltaO-y)])**2/(2*(lambdaDescr*sigma)**2))) * np.linalg.norm([grad_ligne[o-1][s-1][a-1,b-1], grad_col[o-1][s-1][a-1,b-1]])
                        for i in range(1,nhist+1):
                            for j in range(1,nhist+1):
                                fact = 2*lambdaDescr/nhist
                                inv_fact = 1/fact
                                x_i = (i-(1+nhist)/2)*fact
                                y_j = (j-(1+nhist)/2)*fact
                                if((abs(x_i - x_bis) <= fact) and (abs(y_j - y_bis) <= fact)):
                                    for k in range(1,nOri+1):
                                        theta_k = 2*math.pi*(k-1)/nOri
                                        #erifier que mod 2pi
                                        if (abs((theta_k - theta_bis)%2*math.pi)<2*math.pi/nOri):
                                            histo[i-1,j-1,k-1] += (1-inv_fact*abs(x_i - x_bis))*(1-inv_fact*abs(y_j - y_bis))*(1-inv_fact*abs((theta_bis - theta_k)%2*math.pi))*cdescr
                                                    
            for i in range(1,nhist+1):
                 for j in range(1,nhist+1):
                     for k in range(1,nOri+1):
                         feature[(i-1)*nhist*nOri+(j-1)*nOri+k -1]=histo[i-1,j-1,k-1]
            for l in range(1,nhist*nOri*nhist +1):
                feature[l-1] = min(feature[l-1],0.2*np.linalg.norm(feature))
                feature[l-1] = min(int(512*feature[l-1]/np.linalg.norm(feature)),255)  
            outListeFeatured.append([x,y]+feature.tolist())
            
    return np.array(outListeFeatured)

def matriceDeDistance(keyPoints1, keyPoints2) :
    mat = np.zeros((len(keyPoints1),len(keyPoints2)))
    for i, k1 in enumerate(keyPoints1) :
        for j, k2 in enumerate(keyPoints2) :
            desc1 = k1[2:]
            desc2 = k2[2:]
            mat[i,j] = np.linalg.norm(desc2-desc1)
    return mat

def trouverMins(mat, n) :
    coupleAmis = []
    maxi = np.max(mat)
    for i in range(n) :
        mini = np.min(mat)
        ind = np.argwhere(mat==mini)
        coupleAmis.append([ind[0,0],ind[0,1]]) # peut mieux faire, pas optimal
        mat[ind[0,0],ind[0,1]] = maxi
    return coupleAmis
    

    