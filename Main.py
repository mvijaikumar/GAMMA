import sys,math,argparse,os,pdb
import tensorflow as tf
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#np.random.seed(7) ##
#tf.set_random_seed(7)
sys.path.append('./.')
sys.path.append('./Utilities/.')

from Arguments import parse_args
from Parameters import Parameters
from SimpleNegativeSamples import SimpleNegativeSamples as NegativeSamples
from Dataset import Dataset
from ItemView_Dataset import ItemView_Dataset
from Batch import Batch
from Valid_Test_Error import Valid_Test_Error
from Models import Models
from Evaluation import evaluate_model
from Error_plot import Error_plot

from time import time
from pprint import pprint
import random

def get_optimizer(lr,optimizer='adam'):
    if optimizer == 'rmsprop':
        return  tf.compat.v1.train.RMSPropOptimizer(lr)
    elif optimizer == 'adam':
        return  tf.compat.v1.train.AdamOptimizer(lr)

def get_args_to_string(args):
    args_str = str(random.randint(1,1000000))
    return args_str

if __name__ == '__main__':

    args = parse_args()
    print(args)
    args_str      = get_args_to_string(args)
    args.args_str = args_str
    print('Data loading...')
    t1,t_init = time(),time()
    dataset = ItemView_Dataset(args)

    params = Parameters(args,dataset)
    print("""Load data done [%.1f s]. #user:%d, #item:%d, #dom:%d, #train:%d, #test:%d, #valid:%d"""% (time() - t1, params.num_users,
        params.num_items,params.num_doms,params.num_train_instances,params.num_test_instances,params.num_valid_instances))    

    print('Method: %s'%(params.method))
    model = Models(params)    
    model.define_model()
    model.define_loss('all')
    print( "Model definition completed: in %.2fs"%(time()-t1))
           
    train_step = get_optimizer(params.learn_rate,params.optimizer).minimize(model.loss)
    init       = tf.compat.v1.global_variables_initializer()  
    config     = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    print('train instances: {}'.format(params.train_matrix.nnz))

    error_plot = Error_plot(save_flag=True,res_path=params.result_path,args_str=args_str,args=args)
    ns = NegativeSamples(params)
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        vt_err = Valid_Test_Error(params,sess)
        for epoch_num in range(params.num_epochs+1):
            t1 = time()
            user_input,item_input,train_rating,train_domain  = ns.generate_instances(params)
            
            num_inst = len(user_input)
            batch    = Batch(num_inst,params.batch_size,shuffle=True) 
            ce_train, loss,ce_loss,reg_loss,recon_loss = 0.0,0.0,0.0,0.0,0.0
            #print "[{:.2f} s] for negative sampling".format(time()-t1)
            t2 = time()
            while batch.has_next_batch():
                shuff_batch = batch.get_next_batch_indices()
                bsiz        = len(shuff_batch)

                feed_dict_train = {model.user_indices:user_input[shuff_batch],                           
                                       model.item_indices:item_input[shuff_batch],
                                       model.dom_indices:train_domain[shuff_batch],
                                       model.true_rating:train_rating[shuff_batch],
                                       model.keep_prob:params.keep_prob,
                                       model.valid_clip:0.0,
                                       model.batch_siz:bsiz}

                (_,batch_loss,batch_ce_train,batch_reg_err,batch_recon_err) = sess.run([train_step,model.loss,model.ce_loss,model.regularization_loss,
                                                                                        model.recon_error],feed_dict=feed_dict_train)  
                ce_train   += batch_ce_train * bsiz
                
                loss       += batch_loss
                ce_loss    += batch_ce_train
                reg_loss   += batch_reg_err
                recon_loss += batch_recon_err
            ce_train = ce_train/num_inst
            
            batch.initialize_next_epoch()
            print("""[%.2f s] iter:%3i obj ==> total loss:%.4f ce loss:%.4f reg loss:%.4f recon loss:%.4f """
                     %(time()-t2,epoch_num,loss,ce_loss,reg_loss,recon_loss))
            
            # validation and test error
            t3 = time()
            (valid_hits_lst,valid_ndcg_lst) = vt_err.get_update(model,epoch_num,valid_flag=True)
            (test_hits_lst,test_ndcg_lst)   = vt_err.get_update(model,epoch_num,valid_flag=False)
            (valid_hr,valid_ndcg) = (np.mean(valid_hits_lst),np.mean(valid_ndcg_lst))
            (test_hr,test_ndcg)   = (np.mean(test_hits_lst),np.mean(test_ndcg_lst))
            print("[%.2f s] Errors train %.4f valid hr: %.4f test hr: %.4f valid ndcg: %.4f test ndcg: %.4f"%(time()-t3,ce_train,valid_hr,test_hr,valid_ndcg,test_ndcg))
            error_plot.append(loss,recon_loss,reg_loss,ce_loss,valid_hr,test_hr,valid_ndcg,test_ndcg)
        tot_time = time() - t_init
        args.total_time = '{:.2f}m'.format(tot_time/60)
        print 'error plot: '
        (best_valid_hr_index,best_valid_ndcg_index,best_valid_hr,best_valid_ndcg,best_test_hr,best_test_ndcg) = error_plot.get_best_valid_test_error()
        args.hr_index,args.ndcg_index = best_valid_hr_index,best_valid_ndcg_index
        print('[{:.2f} s] best_hr_index: {} best_ndcg_index: {} best_valid_hr: {:.4f} best_valid_ndcg: {:.4f} best_test_hr: {:.4f} best_test_ndcg: {:.4f}'.format(tot_time,best_valid_hr_index,best_valid_ndcg_index,best_valid_hr,best_valid_ndcg,best_test_hr,best_test_ndcg))
        error_plot.plot()
