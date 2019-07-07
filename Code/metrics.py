from data_utils import get_user_genres_list
import numpy as np
#from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
import math
from itertools import chain
from sklearn.metrics import average_precision_score
#np.seterr(invalid='raise')

def populate_user_dict(user_metric_dict,rmd_sz):
    for i in rmd_sz:
        user_metric_dict[i] = dict()
        user_metric_dict[i]['relev_set'] = set()
        user_metric_dict[i]['relev_list'] = list()
        user_metric_dict[i]['dcg'] = dict()
        user_metric_dict[i]['feature_dist'] = dict()
        user_metric_dict[i]['apk'] = dict()
        user_metric_dict[i]['ndcg'] = dict()
        user_metric_dict[i]['cs_prec'] = dict()
###########################################################################################
def populate_result_dict(result_dict,rmd_sz):
    for i in rmd_sz:
        result_dict[i] = dict()
        result_dict[i]['ncall'] = list()
        result_dict[i]['dcg'] = list()
        result_dict[i]['feature_dist'] = list()
        result_dict[i]['apk'] = list()
        result_dict[i]['ndcg'] = list()
        result_dict[i]['cs_prec'] = list()
###########################################################################################
def cs_precision(R_train,rmd_indx):
    return len(np.where(R_train[:,rmd_indx].sum(axis=0) == 0)[0])/float(len(rmd_indx))
###########################################################################################
def ap_K(rmd_ids,relev_rmd_ids):  #mean average precision
    b = np.ones_like(rmd_ids)
    a = np.zeros_like(rmd_ids)
    sorter = np.argsort(rmd_ids)
    a[sorter[np.searchsorted(rmd_ids,relev_rmd_ids,sorter=sorter)]] = 1
    return average_precision_score(a,b) 
###########################################################################################
def serendipity_score(R,u_ind,rmd_indx):
    unrated_test_n = np.where(R[u_ind,rmd_indx] == 0)[0]
    return len(unrated_test_n)/float(len(rmd_indx))
###########################################################################################
def n_call_K(relev_rmd_indx,no):
    return len(relev_rmd_indx)/float(no)
###########################################################################################
def stratified_recall(numr,denom):
    if numr == 0 or denom == 0:
        recall = 0
    else:
        recall = numr/denom
    return recall;
#########################################################################################
def get_user_metrics(user_metric_dict,U_rmd,u_ind,k,R_train,R_test,Mov_Gen_Mat,M,M_map,m_ind,V):
    rmd = U_rmd[u_ind][:k]
    relev_rmd_ids,ratings = relev_rmd_index(R_test,u_ind,rmd,4)
    user_metric_dict[k]['relev_set'].update(relev_rmd_ids)
    user_metric_dict[k]['relev_list'] += list(relev_rmd_ids)
    user_metric_dict[k]['dcg'][u_ind] = dcg(list(rmd),list(relev_rmd_ids))
    user_metric_dict[k]['feature_dist'][u_ind] = average_feature_distance(V,rmd)
    user_metric_dict[k]['apk'][u_ind] = ap_K(list(rmd),list(relev_rmd_ids))
    user_metric_dict[k]['ndcg'][u_ind] = ndcg(list(ratings))
    user_metric_dict[k]['cs_prec'][u_ind] = cs_precision(R_train,rmd)
    return 0;
###########################################################################################
def average_feature_distance(V,rmd_inds):
    #return np.mean(euclidean_distances(V[rmd_inds,:]))
    #print pdist(V[rmd_inds,:],'cosine')
    return np.mean(pdist(V[rmd_inds,:],'cosine'))
###########################################################################################
def true_topic_coverage(R_test,u_ind,rmd_inds,M,M_map,th_score=4):
    gen_list = get_user_genres_list(R_test,u_ind,M,M_map,th_score,m_inds=None)    #relevant genres list, test set
    gen_set = set(chain.from_iterable(gen_list))   #make it set.
    rgen_list = get_user_genres_list(R_test,u_ind,M,M_map,th_score,m_inds=rmd_inds)   #relevant genres list, recommended
    rgen_set = set(chain.from_iterable(rgen_list)) #make it set.
    #print len(rgen_set)
    #print len(gen_set)
    #print "========"
    if len(gen_set) == 0:   #if there is no relevant genres, return 0.
        return 0
    else:
        return float(len(rgen_set))/len(gen_set)
#########################################################################################
def catalog_coverage(numr,denom):
    return float(numr)/denom
#########################################################################################
def relev_rmd_index(R_test,U_ind,rmd_ids,th_score):
    ratings = R_test[U_ind,rmd_ids]
    rel_ind = np.where(R_test[U_ind,rmd_ids] >= th_score)[0]
    return rmd_ids[rel_ind],ratings   #return the relevant item indices in the predicted ratings matrix.
#########################################################################################
def med_pop_rank(R,All_Rmd_Indx,th_score):
    #each column correspond to item.
    tmp_R = np.copy(R)
    tmp_R[tmp_R<th_score] = 0
    tmp_R[tmp_R>=th_score] = 1
    itm_cnts = np.sum(tmp_R,axis=0)    #Get the Item Counts in Obs Data
    rank,orig_indx = np.unique(itm_cnts,return_inverse=True)
    if len(All_Rmd_Indx) == 0:
        return len(rank), 0
    else:
        return len(rank), len(rank)-orig_indx[All_Rmd_Indx]
#########################################################################################
def strat_recall_denm_per_user(R_test,U_ind):
    beta = 0.5 #We set beta value to 0.5
    rel_itms_ind = np.where(R_test[U_ind,:]>=4)[0]   #indexes of items which are relevant for the user in test data
    B = R_test[:,rel_itms_ind] >= 4 #find the number of positive ratings for the relevant items in the observed data.
    B = B.sum(axis = 0)
    B = B[B != 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        denm = np.sum(1.0/np.power(B,beta))
        if math.isinf(denm):
            denm = 0
    return denm
#########################################################################################
def strat_recall_numr_per_user(R_test,U_ind,rmd_inds):
    beta = 0.5
    rel_ind = np.where(R_test[U_ind,rmd_inds] >= 4)[0] #get the relevant indexes from the recommended indexes
    rel_itms_ind = rmd_inds[rel_ind]   #get the relevant indexes
    B = R_test[:,rel_itms_ind] >= 4   #Get the count of relevant recommended items in the test data.create a boolean array
    B = B.sum(axis = 0)    #get the column wise count
    B = B[B != 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        numr = np.sum(1.0/np.power(B,beta))
        if math.isinf(numr):
            numr = 0
    return numr;
#########################################################################################
def dcg(rmd_indx,relev_rmd_indx):
    dcg = [(math.pow(2,rmd_indx[i] in relev_rmd_indx) -1)/math.log(i+2) for i in xrange(len(rmd_indx))]
    return sum(dcg)
###########################################################################################
def ild(Mov_Gen_Mat,rmd_index): #calculated on all the recommended items (relevant and non-relevant) based on the genre vector
    Gen_Mat = Mov_Gen_Mat[rmd_index,:]
    #dist = euclidean_distances(Gen_Mat)
    dist = pdist(Gen_Mat,'cosine')
    return np.mean(dist)
###########################################################################################
def ndcg(relev_scores):
    dcg = [ (math.pow(2,relev_scores[i])-1)/math.log(i+2,2) for i in xrange(len(relev_scores))]
    sorter = sorted(relev_scores,reverse=True)
    nm = [ (math.pow(2,sorter[i])-1)/math.log(i+2,2) for i in xrange(len(sorter))]
    nm = [1 if x == 0 else x for x in nm]
    return sum(dcg)/sum(nm)
###########################################################################################
