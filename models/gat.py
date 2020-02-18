import numpy as np
import tensorflow as tf
import sys,pdb
sys.path.append('./.')
sys.path.append('../utils/.')
sys.path.append('../utils')

from utils import layers
from models.base_gattn import BaseGAttN

class GAT(BaseGAttN):
    def __init__(self):
        pass
    def inference(self,inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, hid_units, n_heads, activation=tf.nn.elu, residual=False):
        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.attn_head(inputs, bias_mat=bias_mat,
                out_sz=hid_units[0], activation=activation,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):
                attns.append(layers.attn_head(h_1, bias_mat=bias_mat,
                    out_sz=hid_units[i], activation=activation,
                    in_drop=ffd_drop, coef_drop=attn_drop, residual=residual))
            h_1 = tf.concat(attns, axis=-1)
            print "h_1",h_1
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.attn_head(h_1, bias_mat=bias_mat,
                out_sz=nb_classes, activation=None,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        #logits = tf.add_n(out) / n_heads[-1]
        ##logits = tf.add_n(out) / n_heads[-1]
        logits = tf.concat(out,axis=-1)
    
        return logits
