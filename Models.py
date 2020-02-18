import tensorflow as tf
import numpy as np
from GAMMA import GAMMA

class Models(object):
    def __init__(self,params):
        self.num_users    = params.num_users
        self.num_items    = params.num_items
        self.num_doms     = params.num_doms
        self.method       = params.method
        self.reg_b        = params.reg_b
        self.reg_w        = params.reg_w
        self.reg_lam      = params.reg_lam

        self.params       = params
        self.initializer  = params.initializer
        
    def define_model(self):
        tf.compat.v1.disable_eager_execution()
        self.user_indices      = tf.compat.v1.placeholder(tf.int32,   shape=[None],name='user_indices')
        self.item_indices      = tf.compat.v1.placeholder(tf.int32,   shape=[None],name='item_indices')
        self.dom_indices       = tf.compat.v1.placeholder(tf.int32,   shape=[None],name='dom_indices')   
        self.keep_prob         = tf.compat.v1.placeholder(tf.float32, name='keep_prob') 
        self.valid_clip        = tf.compat.v1.placeholder(tf.float32, name='valid_clip')
        self.batch_siz         = tf.compat.v1.placeholder(tf.float32, name='batch_siz')

        self.true_rating   = tf.compat.v1.placeholder(tf.float32, shape=[None],name='true_ratings')
                    
        self.model  = GAMMA(self.params)            
        self.model.define_model(self.user_indices,
                                           self.item_indices,
                                           self.dom_indices,
                                           self.true_rating,
                                           self.keep_prob,
                                           self.valid_clip)

        self.pred_rating_model = self.model.pred_rating
        self.pred_rating = self.pred_rating_model
        
    def define_loss(self,loss_type='all'):        
        self.recon_error         = tf.constant(0.0,dtype=tf.float32)
        self.regularization_loss = tf.constant(0.0,dtype=tf.float32)
        self.model.define_loss(loss_type=loss_type)
        self.regularization_loss = self.model.regularization_loss

        self.ce_loss       = tf.compat.v1.losses.log_loss(self.true_rating,self.pred_rating)
        self.loss          = self.ce_loss + self.regularization_loss + self.recon_error 
