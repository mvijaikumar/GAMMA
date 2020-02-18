#bprbranch
import sys
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

class Parameters(object):
    def __init__(self,args,dataset):  
        # method =========================================================
        self.method               = args.method.lower()
        self.args                 = args
        self.result_path          = args.res_path + args.dataset + '/' + args.method + '/'
        self.loss                 = args.loss
        
        # GAT ===========================================================
        self.attn_keep = args.attn_keep
        self.ffd_keep  = args.ffd_keep
        self.proj_keep = args.proj_keep
        self.hid_units = eval(args.hid_units)
        self.n_heads   = eval(args.n_heads)

        # count ===========================================================
        self.num_users            = dataset.num_user
        self.num_items            = dataset.num_item
        self.num_doms             = dataset.num_dom
        self.num_train_instances  = dataset.train_matrix.nnz ##len(dataset.trainArrQuadruplets[0])
        self.num_valid_instances  = len(dataset.validNegativesDict.keys())
        self.num_test_instances   = len(dataset.testNegativesDict.keys())
         
        # data-structures ==================================================
        self.dom_item_dict        = dataset.dom_item_dict
        self.domain_matrix        = dataset.domain_matrix
        self.train_matrix         = dataset.train_matrix
        self.testNegativesDict    = dataset.testNegativesDict   
        self.validNegativesDict   = dataset.validNegativesDict

        self.dom_num_item         = dict()
        for ind in xrange(self.num_doms):
            self.dom_num_item[ind] = len(self.dom_item_dict[ind])
        self.dom_num_item        = dict()
        for ind in xrange(self.num_doms):
            self.dom_num_item[ind] = len(self.dom_item_dict[ind])
        
        # algo-parameters =======================================================
        self.num_epochs            = args.epochs
        self.batch_size            = args.batch_size
        self.valid_batch_siz       = args.valid_batch_siz
        self.learn_rate            = args.lr
        self.optimizer             = args.optimizer
        self.proj_keep             = args.proj_keep
        self.attention_keep        = args.attention_keep
        
        # valid test =======================================================
        self.at_k                  = args.at_k
        self.num_thread            = args.num_thread
        
        # hyper-parameters ======================================================
        self.num_factors           = args.num_factors
        self.num_layers            = args.num_layers ## testing
        self.num_negatives         = args.num_negatives
        self.reg_w                 = args.reg_Wh
        self.reg_b                 = args.reg_bias
        self.reg_lam               = args.reg_lambda
        self.keep_prob             = args.keep_prob
        self.mask_noise            = args.mask_noise #new

        self.num_views             = args.num_views
        self.num_memory_heads      = args.num_memory_heads
        self.memory_head_size      = args.memory_head_size
        self.num_attn_heads        = args.num_attn_heads
        self.attn_head_size        = args.attn_head_size

        # gamma ====================================
        self.adjacency_view_matrix = dataset.adjacency_view_matrix

        # initializations ====================================================================
        if args.initializer == 'xavier':
            print('Initializer: xavier')
            self.initializer = tf.compat.v2.initializers.GlorotNormal()
        elif args.initializer == 'random_normal':
            print('Initializer: random_normal')
            _stddev = args.stddev
            self.initializer = tf.random_normal_initializer(stddev=_stddev)
        elif args.initializer == 'random_uniform':
            print('Initializer: random_uniform')
            _min,_max = -args.stddev, args.stddev
            self.initializer = tf.random_uniform_initializer(minval=_min,maxval=_max)
            
    def get_train_instances(self,train,dom_mat,num_negatives=2):
        user_input, item_input, dom_input, labels = [],[],[],[]
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            dom_input.append(dom_mat[(u, i)] - 10)
            labels.append(1)
            # negative instances
            for t in xrange(num_negatives):
                dom_id   = dom_mat[(u, i)] - 10 # to avoid explicit zero elimination prob
                ind_item = np.random.randint(self.dom_num_item[dom_id])
                j        = self.dom_item_dict[dom_id][ind_item]
                while train.has_key((u, j)):
                    ind_item = np.random.randint(self.dom_num_item[dom_id]) ##new de-bugged
                    j        = self.dom_item_dict[dom_id][ind_item] ##new
                user_input.append(u)
                item_input.append(j)
                dom_input.append(dom_id)
                labels.append(0)
        return np.array(user_input), np.array(item_input), np.array(labels,dtype=np.float16), np.array(dom_input) 
        
    def get_train_instances_old(self,train, num_negatives=2):
        user_input, item_input, labels = [],[],[]
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            labels.append(1)
            # negative instances
            for t in xrange(num_negatives):
                j = np.random.randint(self.num_items)
                while train.has_key((u, j)):
                    j = np.random.randint(self.num_items)
                user_input.append(u)
                item_input.append(j)
                labels.append(0)
        return user_input, item_input, labels
    
    def get_adjacency_matrix_single(self,mat):
        return mat.todense().astype(np.int8)

