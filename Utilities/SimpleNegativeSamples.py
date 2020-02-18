import numpy as np
import sys
from time import time
import scipy.sparse as sp

class SimpleNegativeSamples(object):
    def __init__(self,params):
        # positive part
        (self.user_pos_arr,self.item_pos_arr,self.rating_pos_arr,self.domain_pos_arr) = self.get_positive_instances(params)
        self.num_ratings_per_domain,self.domain_position_in_matrix = self.get_num_ratings_per_domain_and_domain_position(params)

        # negative part
        self.user_neg_arr   = np.repeat(self.user_pos_arr,params.num_negatives)
        self.domain_neg_arr = np.repeat(self.domain_pos_arr,params.num_negatives)
        self.rating_neg_arr = np.repeat([0],len(self.rating_pos_arr) * params.num_negatives)
        
        # positive_and_negative part pre-generated to improve efficiency
        self.user_arr   = np.concatenate([self.user_pos_arr,self.user_neg_arr])
        self.domain_arr = np.concatenate([self.domain_pos_arr,self.domain_neg_arr])
        self.rating_arr = np.concatenate([self.rating_pos_arr,self.rating_neg_arr])
        self.rating_arr = self.rating_arr.astype(np.float16)

    def get_positive_instances(self,params):
        user_pos_arr,item_pos_arr,rating_pos_arr,domain_pos_arr=(np.array([],dtype=np.int),np.array([],dtype=np.int),np.array([],dtype=np.int),np.array([],dtype=np.int))
        domain_mat     = params.domain_matrix.tocsc().tocoo()
        (user_pos_arr,item_pos_arr,domain_pos_arr) = (domain_mat.row,domain_mat.col,domain_mat.data - 10)
        rating_pos_arr = np.repeat([1],len(user_pos_arr))

        return user_pos_arr,item_pos_arr,rating_pos_arr,domain_pos_arr
    
    def get_num_ratings_per_domain_and_domain_position(self,params):
        number_ratings_per_domain = dict()
        domain_position_in_matrix = dict()
        unique_dom, indices = np.unique(self.domain_pos_arr,return_index=True)
        position_arr = unique_dom[np.argsort(indices)]
        print position_arr
        for ind in xrange(len(position_arr)):
            domain_position_in_matrix[ind] = position_arr[ind]

            dom_id = domain_position_in_matrix[ind]
            number_ratings_per_domain[dom_id] = np.sum(self.domain_pos_arr == dom_id)
        
        return number_ratings_per_domain,domain_position_in_matrix

    def generate_negative_item_samples(self,params):
        neg_item_arr = np.array([],dtype=np.int)
        for ind in xrange(len(self.domain_position_in_matrix)):
            dom_id = self.domain_position_in_matrix[ind] 
            if params.dom_num_item[dom_id] != 0:
                if params.method == 'bpr': ## in ['bpr','gbpr']  
                    random_indices = np.random.choice(params.dom_num_item[dom_id], 1 * self.num_ratings_per_domain[dom_id])
                else:
                    random_indices = np.random.choice(params.dom_num_item[dom_id], params.num_negatives * self.num_ratings_per_domain[dom_id])
                neg_items_per_domain = params.dom_item_dict[dom_id][random_indices] 
                neg_item_arr = np.concatenate([neg_item_arr,neg_items_per_domain])

        return neg_item_arr
   
    # call from outside to generate instances at each epochs
    def generate_instances(self,params):
        self.item_neg_arr = self.generate_negative_item_samples(params)
        self.item_arr     = np.concatenate([self.item_pos_arr,self.item_neg_arr])

        return self.user_arr,self.item_arr,self.rating_arr,self.domain_arr
    
    def generate_instances_bpr(self,params):
        self.item_neg_arr = self.generate_negative_item_samples(params)

        return self.user_pos_arr,self.item_pos_arr,self.item_neg_arr,self.domain_pos_arr

