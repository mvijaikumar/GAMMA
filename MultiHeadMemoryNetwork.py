import tensorflow as tf
import numpy as np
import pdb
from Attention import Attention

class MultiHeadMemoryNetwork(object):
    def __init__(self, num_users, attr_dim, initializer, num_memory_heads, num_attn_heads, attn_head_size, memory_head_size):
        
        self.num_users          = num_users
        self.attr_dim           = attr_dim
        self.initializer        = initializer        

        self.num_memory_heads   = num_memory_heads
        self.num_attn_heads     = num_attn_heads
        self.attn_head_size     = attn_head_size
        self.memory_head_size   = memory_head_size
        
        # list initialization ===============================
        self.Key,self.Mem,self.item_attr_mat,self.user_embeddings_view = ([None]*self.num_memory_heads,[None]*self.num_memory_heads,[None]*self.num_memory_heads,[None]*self.num_memory_heads)
        self.item_view_embeds_n,self.cross_user_item = ([None]*self.num_memory_heads,[None]*self.num_memory_heads)
        self.att_key,self.att_mem,self.item_view_mul = ([None] * self.num_memory_heads,[None] * self.num_memory_heads,[None] * self.num_memory_heads)
        self.item_view,self.m3_item_view = ([None] * self.num_memory_heads,[None] * self.num_memory_heads)
        self.user_embeds_view= [None] * self.num_memory_heads

        # ===================================================
        for i in range(self.num_memory_heads):
            self.Key[i]                  = tf.Variable(tf.compat.v1.random_uniform([self.attr_dim, self.memory_head_size], -0.1, 0.1))
            self.Mem[i]                  = tf.Variable(tf.compat.v1.random_uniform([self.memory_head_size, self.attr_dim], 1.0, 1.0))
        self.user_embeddings_view  = tf.Variable(self.initializer(shape=[self.num_users,self.attn_head_size]),dtype=tf.float32,name='user_embeddings_view')    
        self.dense_attn   = tf.keras.layers.Dense(self.attn_head_size,use_bias=True,activation='elu')

    def call(self,user_embeds,user_indices,item_view_embeds, attention_keep): 
        with tf.name_scope("memory_attention"):
            self.uid_n                 = tf.nn.l2_normalize(user_embeds, 1)
            self.user_embeds_view      = tf.nn.embedding_lookup(self.user_embeddings_view, user_indices)             

            for i in range(self.num_memory_heads):
                self.item_view_embeds_n[i]   = tf.nn.l2_normalize(item_view_embeds, 2)
                self.cross_user_item[i]      = tf.einsum('ac,abc->abc', self.uid_n, self.item_view_embeds_n[i])
                self.att_key[i]              = tf.einsum('abc,ck->abk', self.cross_user_item[i], self.Key[i])
                self.att_mem[i]              = tf.nn.softmax(self.att_key[i])

                self.item_view_mul[i]        = tf.einsum('abc,ck->abk', self.att_mem[i], self.Mem[i])
                self.m3_item_view[i]         = tf.multiply(self.item_view_mul[i], item_view_embeds)

            self.mult_cat                    = tf.concat(self.m3_item_view,axis=1)
            #=================

            self.user_embeddings_attn    = tf.Variable(self.initializer(shape=[self.num_users,self.attn_head_size]),dtype=tf.float32,name='user_embedding2')
            self.user_embeds_attn        = tf.nn.embedding_lookup(self.user_embeddings_attn, user_indices)
            self.attn                    = Attention(num_factors=self.attn_head_size,attr_dim=self.attn_head_size,seq_len=self.num_memory_heads,initializer=self.initializer)
            self.attn_weights            = self.attn.get_attention_weights(self.user_embeds_attn,tf.concat(self.m3_item_view,axis=1),atten_type='dotprod') 
            self.attn_weights            = tf.nn.dropout(self.attn_weights, attention_keep)
            self.cont_mat                = self.attn.get_context_matrix(self.attn_weights,tf.concat(self.m3_item_view,axis=1))
            self.attn_mat                = self.attn.get_attention_matrix(self.cont_mat,self.user_embeds_attn,inter_type='context')

            self.final_item_view_output  = self.attn_mat
            self.res = tf.reshape(self.final_item_view_output,shape=[-1,1,self.attn_head_size])
        return  self.res
