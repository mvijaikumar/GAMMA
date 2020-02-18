'''reference : https://github.com/philipperemy/keras-attention-mechanism/issues/14'''
import tensorflow as tf
import numpy as np
import os

class Attention(object):
    def __init__(self, num_factors, attr_dim, seq_len, initializer):
	self.num_factors = num_factors
	self.attr_dim    = attr_dim
        self.seq_len     = seq_len
        self.initializer = initializer

    def get_attention_score(self,mat,item_tensor,atten_type='dotprod'): # 1. Luong's multiplicative style, 2. Bahdanau's additive style
        if atten_type == 'dotprod':
            print 'dotprod'
            mat          = tf.expand_dims(mat,1) #mat to tensor of dim 3 to match item tensor 1 denote axis1
            atten_score  = tf.reduce_sum(tf.multiply(mat,item_tensor),axis=2)
            return atten_score
        elif atten_type == 'weight_vector_dotprod':
            print 'weight_vector_dotprod'
	    self.weight_ten_1  = tf.Variable(self.initializer(shape=[self.num_factors]),dtype=tf.float32,name='weight_ten')
            self.weight_ten    = tf.linalg.diag(self.weight_ten_1)

            mat          = tf.expand_dims(mat,1) #mat to tensor of dim 3 to match item tensor 1 denote axis1
            # matrix-tensor multiplication========
            item_tensor_reshaped = tf.reshape(item_tensor, [-1, self.attr_dim])
            ten_term             = tf.matmul(item_tensor_reshaped, self.weight_ten)
            ten_term             = tf.reshape(ten_term, [-1, self.seq_len, self.num_factors])
            # ====================================
            #atten_score  = tf.reduce_sum(tf.multiply(mat,item_tensor),axis=2)
            atten_score  = tf.reduce_sum(tf.multiply(mat,ten_term),axis=2)
            return atten_score

        elif atten_type == 'weight_dotprod':
            print 'weight_dotprod'
	    self.weight_ten  = tf.Variable(self.initializer(shape=[self.attr_dim,self.num_factors]),dtype=tf.float32,name='weight_ten')
            mat          = tf.expand_dims(mat,1) #mat to tensor of dim 3 to match item tensor 1 denote axis1
            # matrix-tensor multiplication========
            item_tensor_reshaped = tf.reshape(item_tensor, [-1, self.attr_dim])
            ten_term             = tf.matmul(item_tensor_reshaped, self.weight_ten)
            ten_term             = tf.reshape(ten_term, [-1, self.seq_len, self.num_factors])
            # ====================================
            #atten_score  = tf.reduce_sum(tf.multiply(mat,item_tensor),axis=2)
            atten_score  = tf.reduce_sum(tf.multiply(mat,ten_term),axis=2)
            return atten_score

        elif atten_type == 'luong': # Weight allows us to have different dim for user and item_attr vectors
            print 'luong'
	    self.luong_weight = tf.Variable(self.initializer(shape=[self.num_factors,self.attr_dim]),dtype=tf.float32,name='luong_weight')
            mat               = tf.matmul(mat,self.luong_weight)
            mat               = tf.expand_dims(mat,1)
            atten_score       = tf.reduce_sum(tf.multiply(mat,item_tensor),axis=2)
            return atten_score

        elif atten_type == 'bahdanau_old':
            print 'bahdanau_old'
	    self.v_a         = tf.Variable(self.initializer(shape=[1,1,self.num_factors]),dtype=tf.float32,name='bahdanau')
	    self.weight_mat  = tf.Variable(self.initializer(shape=[self.num_factors,self.num_factors]),dtype=tf.float32,name='weight_mat')
	    self.weight_ten  = tf.Variable(self.initializer(shape=[1,self.attr_dim,self.num_factors]),dtype=tf.float32,name='weight_ten')
	    
	    mat_term    = tf.expand_dims(tf.matmul(mat,self.weight_mat),1)
	    ten_term    = tf.reshape(tf.tensordot(item_tensor, self.weight_ten,axes=[2,1]),shape=[item_tensor.shape[0],item_tensor.shape[1],self.weight_ten.shape[2]])
	    sum_term    = mat_term + ten_term
	    atten_score = tf.reduce_sum(tf.multiply(self.v_a, (sum_term)),axis=2)
	    return atten_score

        elif atten_type == 'bahdanau':
            print 'bahdanau new'
	    self.v_a         = tf.Variable(self.initializer(shape=[1,1,self.num_factors]),dtype=tf.float32,name='bahdanau')
	    self.weight_mat  = tf.Variable(self.initializer(shape=[self.num_factors,self.num_factors]),dtype=tf.float32,name='weight_mat')
	    self.weight_ten  = tf.Variable(self.initializer(shape=[self.attr_dim,self.num_factors]),dtype=tf.float32,name='weight_ten')
	    
	    mat_term    = tf.expand_dims(tf.matmul(mat,self.weight_mat),1)
            # matrix-tensor multiplication========
            item_tensor_reshaped = tf.reshape(item_tensor, [-1, self.attr_dim])
            ten_term             = tf.matmul(item_tensor_reshaped, self.weight_ten)
            ten_term             = tf.reshape(ten_term, [-1, self.seq_len, self.num_factors])
            # ====================================
	    sum_term    = mat_term + ten_term
	    #atten_score = tf.reduce_sum(tf.multiply(self.v_a, tf.nn.relu(sum_term)),axis=2)
	    atten_score = tf.reduce_sum(tf.multiply(self.v_a, tf.nn.tanh(sum_term)),axis=2)
	    return atten_score

    def get_attention_weights(self,mat,item_tensor,atten_type='simple'):
        atten_score  = self.get_attention_score(mat,item_tensor,atten_type)
        atten_weight = tf.nn.softmax(atten_score)
        return atten_weight

    def get_context_matrix(self,atten_weight,item_tensor):
        atten_weight = tf.expand_dims(atten_weight,2)
        context_mat  = tf.reduce_sum(tf.multiply(atten_weight,item_tensor),axis=1)
        return context_mat

    def get_attention_matrix(self,context_mat,mat,inter_type='context'):
        if inter_type == 'context': #returns context vector
            print('Context')
            return context_mat
        
        elif inter_type == 'multiply': #shape should be same (simple dot product)
            print('multiply')
            return tf.multiply(context_mat, mat)

	elif inter_type == 'simple_weight':
            print('simple_weight')
            self.proj_weight = tf.Variable(self.initializer(shape=[self.attr_dim,self.num_factors]),dtype=tf.float32,name='cont_weight')
            return tf.multiply(tf.matmul(context_mat,self.proj_weight), mat)
        
        elif inter_type == 'simple_weight_concat':
            print('simple_weight_concat')
            self.cont_weight = tf.Variable(self.initializer(shape=[self.num_factors + self.attr_dim,self.num_factors]),dtype=tf.float32,name='cont_weight')
     	    return tf.matmul(tf.concat([context_mat,mat],axis=1),self.cont_weight)

        elif inter_type == 'tanh': # just for the completion
            print('tanh')
	    self.cont_weight = tf.Variable(self.initializer(shape=[self.num_factors + self.attr_dim,self.num_factors]),dtype=tf.float32,name='cont_weight')
            return tf.nn.tanh(tf.matmul(tf.concat([context_mat,mat],axis=1),self.cont_weight))
