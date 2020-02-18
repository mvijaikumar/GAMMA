import tensorflow as tf
import numpy as np
import pdb,sys
sys.path.append('./.')
sys.path.append('./utils/.')
sys.path.append('./utils')

#############################################################
# The following packages -- models and utils are            #
# adapted from https://github.com/PetarV-/GAT/              #
#############################################################

from models import SpGAT
from utils import process
from MultiHeadMemoryNetwork import MultiHeadMemoryNetwork
from Attention import Attention

class GAMMA(object):
    def __init__(self,params):
        
        self.num_factors            = params.num_factors        
        self.num_users              = params.num_users
        self.num_items              = params.num_items  
        self.num_doms               = params.num_doms
        self.attr_dim               = params.attn_head_size
        self.reg_lam                = params.reg_lam
        self.reg_w                  = params.reg_w
        self.reg_b                  = params.reg_b
        self.initializer            = params.initializer        

        self.num_views              = params.num_views
        self.num_attn_heads         = params.num_attn_heads
        self.num_memory_heads       = params.num_memory_heads
        self.attn_head_size         = params.attn_head_size
        self.memory_head_size       = params.memory_head_size
        self.proj_keep              = params.proj_keep
        self.attention_keep         = params.attention_keep
        self.params                 = params

        # list initializtion ====================================
        self.memnet                 = [None] * self.num_views
        self.mult_views             = [None] * self.num_views
        self.item_view_embeds       = [None] * self.num_views
        self.item_attr_mat          = [None] * self.num_views
        self.user_embeds_view       = [None] * self.num_views
        self.proj_item_view         = [None] * self.num_views
        self.mhead_item_view_output = [None] * self.num_views
        self.m3_item_view_output    = [None] * self.num_views

        # gat ====
        self.model                  = [None] * self.num_views
        self.biases                 = [None] * self.num_views
        self.num_nodes              = [None] * self.num_views
        self.X_features_view        = [None] * self.num_views
        self.bias_in                = [None] * self.num_views
        self.logits                 = [None] * self.num_views
        self.item_view_embeddings   = [None] * self.num_views
        self.item_view_embeds       = [None] * self.num_views
        self.X_features_item_entity_view = [None] * self.num_views

        # =====================================================

        self.adjacency_view_matrix  = params.adjacency_view_matrix
        for view in range(self.num_views):
            self.model[view]        = SpGAT()
            self.biases[view]       = process.preprocess_adj_bias(params.adjacency_view_matrix[view])
            self.num_nodes[view]    = params.adjacency_view_matrix[view].shape[0] 

        self.out_size               = params.num_factors 
        self.hid_units              = params.hid_units 
        self.n_heads                = params.n_heads 
        self.attn_keep              = params.attn_keep 
        self.ffd_keep               = params.ffd_keep 
        self.proj_keep              = params.proj_keep 
        self.residual               = False
        self.nonlinearity           = tf.nn.elu
        self.dense                  = tf.keras.layers.Dense(self.attn_head_size,use_bias=True,activation='elu')
        self.dense_attn             = tf.keras.layers.Dense(self.attn_head_size/2,use_bias=True,activation='elu')
        self.dense_w                = tf.keras.layers.Dense(1,use_bias=False,activation='sigmoid')

    def define_model(self,user_indices,item_indices,dom_indices,true_rating,keep_prob,valid_clip):
        self.user_indices           = user_indices
        self.item_indices           = item_indices
        self.dom_indices            = dom_indices
        self.true_rating            = true_rating
        self.keep_prob              = keep_prob  
        self.valid_clip             = valid_clip
        self.is_training            = tf.equal(0.0,valid_clip)
        
        # variables
        self.user_embeddings        = tf.Variable(self.initializer(shape=[self.num_users,self.num_factors]),dtype=tf.float32,name='user_embedding')
        self.item_embeddings        = tf.Variable(self.initializer(shape=[self.num_items,self.num_factors]),dtype=tf.float32,name='item_embedding')
        # definitions
        self.user_embeds            = tf.nn.embedding_lookup(self.user_embeddings, self.user_indices)
        self.item_embeds            = tf.nn.embedding_lookup(self.item_embeddings, self.item_indices)
        
        self.multiplied_output1     = tf.multiply(self.user_embeds,self.item_embeds)
        self.multiplied_output1     = tf.nn.dropout(self.multiplied_output1, self.keep_prob)

        for view in range(self.num_views):
            # gat layer ============================
            self.X_features_item_entity_view[view] = tf.Variable(self.initializer(shape=[self.num_nodes[view],self.num_factors]),trainable=True,dtype=tf.float32,name='X_features_item_entity')
            self.X_features_view[view]             = tf.expand_dims(self.X_features_item_entity_view[view],0)
            self.bias_in[view]                     = tf.SparseTensor(indices=self.biases[view][0],values=self.biases[view][1],dense_shape=self.biases[view][2])
            self.logits[view]                      = self.model[view].inference(self.X_features_view[view], self.out_size, nb_nodes=self.num_nodes[view], training=self.is_training,
                                                     attn_drop = (1-self.attn_keep) * (1-self.valid_clip), ffd_drop = (1-self.ffd_keep) * (1-self.valid_clip),
                                                     bias_mat=self.bias_in[view],
                                                     hid_units=self.hid_units, n_heads=self.n_heads,
                                                     residual=self.residual, activation=self.nonlinearity)

            self.item_view_embeddings[view]        = tf.reshape(self.logits[view], [-1, self.n_heads[-1] * self.out_size])         
            self.item_view_embeds[view]            = tf.nn.embedding_lookup(self.item_view_embeddings[view], self.item_indices)

            self.proj_item_view[view]              = self.dense(self.item_view_embeds[view])
            self.proj_item_view[view]              = tf.nn.dropout(self.proj_item_view[view], self.keep_prob)
            
            # m3 layer ========================
            self.memnet[view]                      = MultiHeadMemoryNetwork(num_users=self.num_users,attr_dim=self.attr_dim,initializer=self.initializer, 
                                                                            num_memory_heads=self.num_memory_heads,num_attn_heads=self.num_attn_heads,
                                                                            attn_head_size=self.attn_head_size,memory_head_size=self.memory_head_size)
            self.item_indices_expanded             = tf.expand_dims(item_indices,1)
            self.m3_item_view_output[view]         = self.memnet[view].call(user_embeds=self.user_embeds,user_indices=self.user_indices,
                                                                                     item_view_embeds=tf.expand_dims(self.proj_item_view[view],1),
                                                                                     attention_keep=self.valid_clip + self.attention_keep * (1-self.valid_clip))
        # Attention ==================
        self.user_embeddings_attn                  = tf.Variable(self.initializer(shape=[self.num_users,self.attn_head_size]),dtype=tf.float32,name='user_embedding2')
        self.user_embeds_attn                      = tf.nn.embedding_lookup(self.user_embeddings_attn, self.user_indices)
        self.attn                                  = Attention(num_factors=self.num_factors,attr_dim=self.attn_head_size,seq_len=self.num_views,initializer=self.initializer)
        #self.attn_weights                          = self.attn.get_attention_weights(self.user_embeds_attn,tf.concat(self.m3_item_view_output,axis=1),atten_type='bahdanau') 
        self.attn_weights                          = self.attn.get_attention_weights(self.user_embeds_attn,tf.concat(self.m3_item_view_output,axis=1),atten_type='luong') 
        self.attn_weights                          = tf.nn.dropout(self.attn_weights, self.valid_clip + self.attention_keep * (1-self.valid_clip))
        self.cont_mat                              = self.attn.get_context_matrix(self.attn_weights,tf.concat(self.m3_item_view_output,axis=1))
        self.attn_mat                              = self.attn.get_attention_matrix(self.cont_mat,self.user_embeds_attn,inter_type='context')

        self.final_item_view_output                = self.attn_mat
        self.final_item_view_output                = self.dense_attn(self.final_item_view_output)

        # ============================
        self.user_embeddings_attr                  = tf.Variable(self.initializer(shape=[self.num_users,self.attn_head_size/2]),dtype=tf.float32,name='user_embedding2')
        self.user_embeds_attr                      = tf.nn.embedding_lookup(self.user_embeddings_attr, self.user_indices)

        self.multiplied_output2                    = tf.multiply(self.user_embeds_attr,tf.reshape(self.final_item_view_output,shape=[-1,self.attn_head_size/2])) ##
        self.multiplied_output2                    = tf.nn.dropout(self.multiplied_output2, self.valid_clip + self.proj_keep * (1-self.valid_clip))

        # ============================
        self.mult_cat                              = tf.concat([self.multiplied_output1,self.multiplied_output2],axis=1) ##2
        self.pred_rating                           = tf.reshape(self.dense_w(self.mult_cat),shape=[-1])
        
    def define_loss(self,loss_type='all'):
        self.regularization_loss = tf.constant(0.0)
