#!/usr/bin/python2
import numpy as np
from os.path import isfile
from numpy.linalg import norm,lstsq
from scipy.optimize import nnls
import logging
from sklearn.externals.joblib import Parallel,delayed
from sklearn.cluster import k_means
from sklearn.preprocessing import LabelBinarizer

logger = logging.getLogger(__name__)

def cmf(R,X,folder_path,splt,rank, max_iter=50, eps=1e-5, lmbda = 0.01,mu = 0.01):   #matrix and dictionary factorization
    #Here R is the augmented matrixx augmented row wise or column wise.
    uf_file = folder_path + "cU_%d.txt" %(splt)
    ff_file = folder_path + "cF_%d.txt" %(splt)
    gf_file = folder_path + "cG_%d.txt" %(splt)
    if isfile(uf_file) and isfile(gf_file): 
        logging.info("Loading already generated feature and parameter matrices")
        U = np.loadtxt(uf_file,dtype=np.float32,delimiter=',')
        #F = np.loadtxt(ff_file,dtype=np.float32,delimiter=',')
        G = np.loadtxt(gf_file,dtype=np.float32,delimiter=',')
        return U,G
    #run K-means to get an initial estimation of G
    random = np.random.RandomState(seed=splt)
    F,labels,_ = k_means(X,n_clusters=rank,random_state=random,n_jobs=10,n_init=1)
    lb = LabelBinarizer()
    G = lb.fit_transform(labels).astype(np.float32)
    G += 0.2	#add 0.2 to make it perfect soft clusters
    pObj = 0
    no_imp = 0
    no_s = X.shape[1]
    #F = random.rand(rank,no_s)
    for i in xrange(1,max_iter+1):
        topF = np.dot(G.T,X)
        #if np.isnan(topF).any(): #    print "TopF nan"; print topF; print botF #    print topG #    print botG #    exit()
        botF = np.dot(np.dot(G.T,G),F)
        #if np.isnan(botF).any(): #    print "BotF nan" #    print topF #    print botF #    print topG #    print botG #    exit()
        #F *= topF/botF
        F *= np.divide(topF,botF,out=np.zeros_like(topF), where=botF!=0)
        topG = np.dot(X,F.T)
        #if np.isnan(topG).any(): #    print "TopG nan"        #print topF        #print botF        #print topG        #print botG#    #exit()
        botG = np.dot(G,np.dot(F,F.T))
        #if np.isnan(botG).any(): #    print "BotG nan"#    print topF#    print botF#    print topG#    print botG#    exit()
        #G *= topG / botG
        G *= np.divide(topG,botG,out=np.zeros_like(topG), where=botG!=0)
        Obj = clus_obj_func(X,G,F)
        logger.info("Cluster Objective Function After %d Iteration: %f\n" %(i,Obj))
        if (abs(Obj-pObj) < eps):
            no_imp += 1;
        else:   
            no_imp = 0
        if ((abs(Obj-pObj) < eps) and (no_imp >= 10)):
            logger.info("No Improvement in objective function for 10 continuous iterations.  Exiting..... \n")
            break
        pObj = Obj
    #once we get the soft cluster membership, we need to find the corresponding user feature
    W = np.sign(R)
    rows, columns = R.shape
    Ir = np.eye(rank)
    logger.info("Solving for cU\n")
    #U = Parallel(n_jobs=-1)(delayed(lstsq)(np.dot(G.T,np.dot(np.diag(Wu),G))+lmbda*Ir, np.dot(R[u],np.dot(np.diag(Wu),G)),rcond=None) for u,Wu in enumerate(W))
    U = Parallel(n_jobs=-1)(delayed(nnls)(np.dot(G.T,np.dot(np.diag(Wu),G))+lmbda*Ir, np.dot(R[u],np.dot(np.diag(Wu),G))) for u,Wu in enumerate(W))
    U = np.vstack([u[0] for u in U])
    write_U_mat(folder_path,U,splt)
    write_G_mat(folder_path,G,splt)
    write_F_mat(folder_path,F,splt)
    return U,G

def write_U_mat(folder_path,U,splt):
    logging.info('Writing cU matrix to the file')
    U_file = folder_path + "cU_%d.txt" %(splt)
    np.savetxt(U_file,U,delimiter=',')

def write_G_mat(folder_path,G,splt):
    logging.info('Writing cG matrix to the file')
    G_file = folder_path + "cG_%d.txt" %(splt)
    np.savetxt(G_file,G,delimiter=',')

def write_F_mat(folder_path,F,splt):
    logging.info('Writing cF matrix to the file')
    F_file = folder_path + "cF_%d.txt" %(splt)
    np.savetxt(F_file,F,delimiter=',')

def clus_obj_func(X,G,F):
    return norm(X-np.dot(G,F))**2
