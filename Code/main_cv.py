#!/usr/bin/env python2.7

import sys
import os.path as path
import numpy as np
from time import time
import logging
from metrics import get_user_metrics,populate_result_dict,populate_user_dict,n_call_K
from utils import get_recommendation
from joblib import Parallel,delayed,cpu_count
from data_utils import *
from sklearn.metrics.pairwise import *
from sklearn.model_selection import KFold
from cmf import cmf

np.seterr(invalid='raise')

def main(alg,rank=25,tot_itr=5): #main function
    max_rat = 5 #maximum rating value
    R = U = V = D = MG = R_est = None
    rmd_sz = [3,5,10,20]  #metrics will be calculated for this recommendation.
    K = max(rmd_sz)
    all_datasets = ['dataset'+str(_) for _ in xrange(1,3)]
    random = np.random.RandomState(seed=tot_itr)
    for dataset in all_datasets:
        result_dict = dict()
        populate_result_dict(result_dict,rmd_sz)    #place-holder for each split results

        folder_path = get_folder_path(dataset)
        R,U_map,M_map = get_rating_matrix(dataset) #U/M_map contains original user/movie id.
        genres = get_genres(dataset)    #we call it topics instead of genres
        (r_sz,c_sz) = R.shape
        M = get_movies(dataset,M_map)   #Load the movies file in a structured array.
        MG = build_genre_mat(M,genres)   #Genre Matrix for the movies. It is a binary matrix of M*18 size. we call it topic matrix
        kf = KFold(n_splits=tot_itr,random_state=random)
        splt = 0
        for _,te_ind in kf.split(xrange(R.shape[1])):    #Instead of 5 random split we use 5 fold cross validation
            splt += 1
            log_file = folder_path + '%s_access_%d.log' %(alg,splt)
            logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
            logging.info("Populating Dictionaries to store the results")
            logging.info("%d KFold Cross-Validation" %(tot_itr))
            U_rmd = dict()
            user_metric_dict = dict()
            populate_user_dict(user_metric_dict,rmd_sz) #place holder for user results
            R_train,R_test = get_KFold_split(R,te_ind) #5% is test set.
            users = get_test_users(R_test)  #get only users who has test ratings.
            if alg == 'cmf':
                #X = np.dot(MG.T,MG).astype(np.float32)
                #X = X/X.sum()
                #X =  np.dot(MG,pinv(np.eye(MG.T.shape[0])-np.dot(MG.T,MG))-np.eye(MG.T.shape[0]))
                X =  np.dot(MG,np.dot(MG.T,MG).astype(np.float32))
                #X = random.randint(0,2,size=(R_train.shape[1],18))
                U,V = cmf(R_train,X,folder_path,splt,rank)
                Rhat = np.dot(U,V.T)
            for u_ind in users:
                m_ind,m_rat = get_rated_movies(R_train,u_ind)   #get already rated movie indxs and ratings values;
                Unrated_M = np.setdiff1d(np.arange(c_sz,dtype=np.uint16),m_ind,assume_unique=True)
                #U_rmd[u_ind] = baselines[alg](u_ind,Unrated_M,Rhat,K);	#Call the function
                U_rmd[u_ind] = get_recommendation(u_ind,Unrated_M,Rhat,K);	#Call the function

                for i in rmd_sz:
                    get_user_metrics(user_metric_dict,U_rmd,u_ind,i,R_train,R_test,MG,M,M_map,m_ind,V)
            del R_train
            del R_test
            logging.info("============================================================================")
            for i in rmd_sz:
                logging.info("Top %d Recommendations" %(i))
                result_dict[i]['ncall'].append(n_call_K(user_metric_dict[i]['relev_list'],len(users)))
                result_dict[i]['dcg'].append(np.mean(user_metric_dict[i]['dcg'].values()))
                logging.info("Average DCG : %f" %(result_dict[i]['dcg'][-1]))
                result_dict[i]['feature_dist'].append(np.mean(user_metric_dict[i]['feature_dist'].values()))
                logging.info("Average Feature Distance : %f" %(result_dict[i]['feature_dist'][-1]))
                result_dict[i]['apk'].append(np.mean(user_metric_dict[i]['apk'].values()))
                logging.info("AP@K : %f" %(result_dict[i]['apk'][-1]))
                result_dict[i]['ndcg'].append(np.mean(user_metric_dict[i]['ndcg'].values()))
                logging.info("NDCG : %f" %(result_dict[i]['ndcg'][-1]))
                result_dict[i]['cs_prec'].append(np.mean(user_metric_dict[i]['cs_prec'].values()))
                logging.info("CS Precision : %f" %(result_dict[i]['cs_prec'][-1]))
            logger = logging.getLogger()
            for hdlr in logger.handlers[:]:
                hdlr.close()
                logger.removeHandler(hdlr)                              
        #here put the code to do average over the total iterations
        for i in rmd_sz:
            print "============================================================================"
            print "============================================================================"
            print "%d recommendations for %s algorithm (%d iterations)" %(i,alg,tot_itr)
            print "Averag n-call@K: %f" %(np.mean(result_dict[i]['ncall']))
            print "Averag DCG: %f" %(np.mean(result_dict[i]['dcg']))
            print "Averag Feature Distance: %f" %(np.mean(result_dict[i]['feature_dist']))
            print "MAP: %f" %(np.mean(result_dict[i]['apk']))
            print "NDCG: %f" %(np.mean(result_dict[i]['ndcg']))
            print "CS Precision: %f" %(np.mean(result_dict[i]['cs_prec']))
    
if __name__ == '__main__':
    baselines = {
            'cmf'   : cmf,  #cluster based cold start recommendations
    }
    #Parallel(n_jobs=cpu_count())(delayed(main)(alg) for alg in baselines.keys())
    for alg in baselines.keys():
        main(alg)
