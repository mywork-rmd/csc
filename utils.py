import numpy as np;
from time import time;
import logging;

logger = logging.getLogger(__name__)

#################################################################################################
def dbcr(u_ind,Unrated_M,Rhat,K): #Dictionary Based ColdStart Recommendations
    if (Unrated_M.size <= K):
        K = Unrated_M.size
    logger.info('DBCR algorithm: ')
    eps = 1e-8
    Rec_Movies_ID = list() #Greedy Set
    indxs = np.argsort(Rhat[u_ind,Unrated_M])[::-1][:K]
    logger.info('Finished DBCR cold-recommendation algorithm')
    return Unrated_M[indxs]    #this gives the item ids
###########################################################################################
def csb(u_ind,Unrated_M,Rhat,K): #Dictionary Based ColdStart Recommendations
    if (Unrated_M.size <= K):
        K = Unrated_M.size
    logger.info('DBCR algorithm: ')
    eps = 1e-8
    Rec_Movies_ID = list() #Greedy Set
    indxs = np.argsort(Rhat[u_ind,Unrated_M])[::-1][:K]
    logger.info('Finished DBCR cold-recommendation algorithm')
    return Unrated_M[indxs]    #this gives the item ids
###########################################################################################
def get_recommendation(u_ind,Unrated_M,Rhat,K):
    if (Unrated_M.size <= K):
        K = Unrated_M.size
    eps = 1e-8
    Rec_Movies_ID = list()
    indxs = np.argsort(Rhat[u_ind,Unrated_M])[::-1][:K]
    return Unrated_M[indxs]    #this gives the item ids
###########################################################################################
