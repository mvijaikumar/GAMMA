#import tensorflow as tf
import numpy as np
import sys
import math

from itertools import chain
from Batch import Batch
from Evaluation import evaluate_model

class Valid_Test_Error:
    def __init__(self,params,sess):
        
        self.validNegativesDict   = params.validNegativesDict
        self.testNegativesDict    = params.testNegativesDict   
        
        self.num_valid_instances  = params.num_valid_instances
        self.num_test_instances   = params.num_test_instances
        self.batch_size           = params.batch_size

        self.epoch_mod       = 1
        self.params          = params
        self.sess            = sess
        self.valid_batch_siz = params.valid_batch_siz
        
        self.at_k        = params.at_k
        self.num_thread  = params.num_thread
        
        self.best_mae, self.best_iter     = 100,-1
        self.best_mse, self.best_mse_iter = 200,-1
        self.history_test_mae  = []
        self.history_valid_mae = []
        
        # init functions
        self.get_num_valid_negative_samples()
        self.valid_dim = self.num_valid_negatives + 1
        
        self.validArrTriplets,self.valid_pos_items = self.get_dict_to_triplets(self.validNegativesDict)
        self.testArrTriplets,self.test_pos_items   = self.get_dict_to_triplets(self.testNegativesDict)
        
    def get_num_valid_negative_samples(self):
        first_key = next(iter(self.validNegativesDict))
        self.num_valid_negatives = len(self.validNegativesDict[first_key])
        
    def get_dict_to_triplets(self,dct):
        user_lst, item_lst, domain_lst = [],[],[]
        pos_item_lst = []
        for key,value in dct.iteritems():
            usr_id, itm_id, dom_id = key
            users = list(np.full(self.valid_dim,usr_id,dtype = 'int32'))#+1 to add pos item
            items = [[itm_id]]
            pos_item_lst.append(itm_id)
            items.append(list(value)) # first is positive item
            items = list(chain.from_iterable(items))
            doms  = list(np.full(self.valid_dim, dom_id, dtype = 'int32'))
        
            user_lst.append(users)
            item_lst.append(items)
            domain_lst.append(doms)
            
        user_lst   = list(chain.from_iterable(user_lst))
        item_lst   = map(int,list(chain.from_iterable(item_lst)))
        domain_lst = list(chain.from_iterable(domain_lst))
            
        return (np.array(user_lst),np.array(item_lst),np.array(domain_lst)),np.array(pos_item_lst)        
                
    def get_update(self,model,epoch_num,valid_flag=True):
        
        if valid_flag == True:
            (user_input,item_input,domain) = self.validArrTriplets
            num_inst   = self.params.num_valid_instances * self.valid_dim
            posItemlst = self.valid_pos_items # parameter for evaluate_model
            matShape   = (self.params.num_valid_instances, self.valid_dim)
            
        else:
            (user_input,item_input,domain) = self.testArrTriplets
            num_inst   = self.params.num_test_instances * self.valid_dim
            posItemlst = self.test_pos_items # parameter for evaluate_model
            matShape   = (self.params.num_test_instances, self.valid_dim)
            
        batch_siz   = self.valid_batch_siz * self.valid_dim
        batch       = Batch(num_inst,batch_siz,shuffle=False) 

        full_pred_lst = []
        while batch.has_next_batch():
            batch_indices = batch.get_next_batch_indices()
            bsiz          = len(batch_indices)
            feed_dict     = {model.user_indices:user_input[batch_indices],                           
                             model.item_indices:item_input[batch_indices],
                             model.dom_indices:domain[batch_indices],
                             model.keep_prob:1.0,
                             model.valid_clip:1.0}
            
            pred_lst = self.sess.run(model.pred_rating,feed_dict=feed_dict) 
            
            if self.params.method.lower() == 'bpr':
                full_pred_lst += list(pred_lst)
            else:
                full_pred_lst += list(pred_lst)
        
        predMatrix    = np.array(full_pred_lst).reshape(matShape) # parameter for evaluate_model
        itemMatrix    = np.array(item_input).reshape(matShape)    # parameter for evaluate_model
        
        (hits, ndcgs) = evaluate_model(posItemlst=posItemlst,itemMatrix=itemMatrix,
                                       predMatrix=predMatrix,k=self.at_k, 
                                       num_thread=self.num_thread)
        return (hits, ndcgs)
