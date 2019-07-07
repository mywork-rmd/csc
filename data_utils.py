#!/usr/bin/python2
import numpy as np
from os.path import isfile
import logging
from sklearn.metrics.pairwise import cosine_similarity
import logging
import math

logger = logging.getLogger(__name__)

dataset_dtls = {
                'dataset1':{
                    'folder_path'   : '../Datasets/MovieLens/',
                    'ratings_file' :   'ratings.dat',
                    'movies_file' : 'movies.dat',
                    'users_file'  : 'users.dat',
                    'genres'    : ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                    },
                'dataset2':{
                    'folder_path'   : '../Datasets/Ymovies/',
                    'ratings_file' :   'ratings_g15.dat',
                    'movies_file' : 'movies_g15.dat',
                    'genres'    :   ['Action', 'Adult Audience', 'Adventure', 'Animation', 'Art', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Foreign', 'Gangster', 'Horror', 'Kids', 'Miscellaneous', 'Musical', 'Performing Arts', 'Reality', 'Romance', 'Science Fiction', 'Special Interest', 'Suspense', 'Thriller', 'Western'],
                    },
                }

###########################################################################################
def get_folder_path(dataset):
    return dataset_dtls[dataset]['folder_path']
###########################################################################################
def get_genres(dataset):
    return dataset_dtls[dataset]['genres']
###########################################################################################
def get_log_file(dataset):
    return dataset_dtls[dataset]['folder_path'] + "access_%d.log" %(source,target,cnt)
###########################################################################################
def get_rating_matrix(dataset):
    ratings_file = dataset_dtls[dataset]['folder_path'] + dataset_dtls[dataset]['ratings_file']
    if isfile(ratings_file):
        print "Reading from the ratings file %s" %(ratings_file)  
        A = np.genfromtxt(ratings_file,delimiter='::',dtype=(np.uint16,np.uint32,np.uint8),names="UID,MID,RATING",usecols=(0,1,2));
        U_map,U_inv = np.unique(A['UID'],return_inverse=True);    #User map
        M_map,M_inv = np.unique(A['MID'],return_inverse=True);   #Movie map
        R = np.zeros((U_map.shape[0],M_map.shape[0]),dtype=np.uint8);    #create user*movie rating matrix. note that this matrix is only for the users and movies which are present in the rating data. so it wont tally with the actual user/movie matrix.
        R[U_inv,M_inv] = A['RATING']; #user-item rating matrix;
    else:
        print "Rating file %s does NOT exist. Exiting " %(ratings_file)
        exit(0)
    return (R,U_map,M_map); #Index of the u_id/m_id in U_map/M_map is the position in r_matrix.which(U_map==ID)
###########################################################################################
def get_movies(dataset,M_map):
    movies_file = dataset_dtls[dataset]['folder_path'] + dataset_dtls[dataset]['movies_file']
    if isfile(movies_file):
        M = np.genfromtxt(movies_file, delimiter='::', dtype=(int,"|S32","|S64"), usecols=(0,1,2),names="ID,NAME,GENRE",encoding='latin1',comments=None)  #create a structured array. easy to do mapping.
        ind = np.where(M['ID'] == M_map.reshape(M_map.shape[0],1))[1]
        M = M[ind]
        return M
    else:
        print "Movies file %s does not exist. It is required to calculate topic coverage" %(movies_file)
        print "Exiting"
        exit(0) 
###########################################################################################
def get_users(dataset):
    users_file = dataset_dtls[dataset]['folder_path'] + dataset_dtls[dataset]['users_file']
    if isfile(users_file):
        U = np.genfromtxt(users_file,delimiter='::',dtype=(np.int32,"|S1",np.int8,np.int8),names="ID,SEX,AGE,JOB,",usecols=(0,1,2,3))
        return U
    else:
        return None
###########################################################################################
def get_test_users(R_test):
    return np.where(R_test.sum(1)!=0)[0]    #list of users who has at least one rating in the test data
###########################################################################################
def get_unrated_movies(R_test,u_ind):
    return np.where(R_test[u_ind,:] != 0)[0]
###########################################################################################
def read_userlist(dataset):
    if 'users' in dataset_dtls[dataset].keys():
        return dataset_dtls[dataset]['users']
    else:
        return None
###########################################################################################
def build_genre_mat(M,genres):
    gen_list = np.core.defchararray.split(M['GENRE'],sep='|')
    A = [[item in itm for item in genres] for itm in gen_list]
    A = np.array(A,dtype=np.float32)
    return A
#########################################################################################
def get_KFold_split(R,Te_Ind):
   R_train = np.copy(R)
   R_test = np.zeros(shape=R.shape)
   R_test[:,Te_Ind] = R[:,Te_Ind]
   R_train[:,Te_Ind] = 0
   return R_train,R_test
#########################################################################################
def split_train_test(R,perc,splt):    #split the data into training and test by random sampling.
    random = np.random.RandomState(seed=splt)
    U,I = R.shape
    k = int(math.ceil(perc*I/100))  #5% test data.
    Indx = random.choice(xrange(I),k,replace=False)
    R_test = np.zeros(shape=(U,I))
    R_train = np.copy(R)
    if len(Indx) == 0:
        return R_train,R_test
    R_test[:,Indx] = R[:,Indx]
    R_train[:,Indx] = 0
    return R_train,R_test
#########################################################################################
def get_item_item_sim(V):  #get the
    D = cosine_similarity(V)
    return D
###########################################################################################
def get_predicted_ratings(U,V,max_rat):
    Rhat = np.dot(U,V)
    #W = float(max_rat)/np.max(Rhat,axis=1)[:,None].astype(float)
    #Rhat -= np.min(Rhat)
    #Rhat = Rhat*W
    return Rhat
#########################################################################################
def get_rated_movies(R,u_ind):   #Given a user indexd it retrieves the movies and ratings by the user.
    m_ind = R[u_ind,:].nonzero()[0]    #get index of already rated movies by user at u_ind. A single Row
    m_rat = R[u_ind,m_ind] #get the ratings of the movies already rated by user at u_ind
    return m_ind,m_rat  #return the already rated movie indexes and corresponding ratings of user id
###########################################################################################
def get_user_genres_list(R,u_ind,M,M_map,th_score,m_inds=None):
    if m_inds is None:  #m_inds is None
        rel_ind = np.where(R[u_ind,:] >= th_score)[0]
    else:
        rel_ind = np.where(R[u_ind,m_inds] >= th_score)[0]
    if len(rel_ind) == 0:
        return list()
    if m_inds is None:
        rel_itms_ind = rel_ind
    else:
        rel_itms_ind = m_inds[rel_ind] #actual relevant indexes from the m_inds
    M_id = M_map[rel_itms_ind] #relevant test movie ids.
    M_ind = np.where(M['ID'] == (M_id.reshape(M_id.shape[0],1)))[1]
    Genr = M['GENRE'][M_ind]   #relevant movie genres, test
    gen_list = np.core.defchararray.split(Genr,sep='|')    #relevant genres list, training
    return gen_list
