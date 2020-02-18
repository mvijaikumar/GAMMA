import numpy as np
import scipy.sparse as sp
from time import time
from collections import defaultdict

class Dataset(object):
    def __init__(self,args):
        dirname  = args.path
        filename = args.dataset
        #self.max_item_seq_length = args.max_item_seq_length
        #self.num_itemidfiles     = args.num_itemidfiles
        self.method = args.method

        self.path     = dirname + filename

        path_valid_test = dirname + filename
        print("filepath: " + self.path)
        
        self.get_user_item_count(self.path + ".train")
        self.embed_path          = self.path #self.get_embed_path(self.path,args.dataset)
        self.trainArrQuadruplets = self.load_rating_file_as_arraylist(self.path + ".train",train_flag=True)
        
        self.validNegativesDict  = self.load_file_as_dict(path_valid_test + ".valid") #.valid.negative
        self.testNegativesDict   = self.load_file_as_dict(path_valid_test + ".test") #.test.negative
        
        (self.train_matrix, self.domain_matrix) = self.load_rating_file_as_matrix(self.path + ".train")
            
    def get_embed_path(self,path,dataset):
        path_words = path.replace('//','/').split('/')
        embed_path = '/'.join(path_words[:-3]) + '/' + dataset

        return embed_path
        
    def get_item_embed_dim(self,filename):
        item_embed = dict()
        with open(filename, "r") as f:
            line = f.readline().strip()
            toks = line.replace("\n","").split("::")
            itemid = int(toks[0])
            #print('\n\n'+line+'\n\n\n\n')
            #print(toks[1]+'\n\n')
            embed  = np.array(toks[1].split(" ")).astype(np.float)
            attr_dim = len(embed)
            return attr_dim
        
    def get_user_item_count(self, filename):
        num_users, num_items, num_domains = 0, 0, 0
        with open(filename, "r") as f:
            line = f.readline().strip()
            while line != None and line != "":
                arr         = line.split("\t")
                u, i, d     = int(arr[0]), int(arr[1]), int(arr[3])
                num_users   = max(num_users, u)
                num_items   = max(num_items, i)
                num_domains = max(num_domains, d)
                line = f.readline()
        self.num_user   = num_users+1
        self.num_item   = num_items+1
        self.num_dom    = num_domains+1
    
    def load_rating_file_as_arraylist(self, filename,train_flag=False):
        user_input, item_input, rating, domain = [],[],[],[]
        
        if train_flag:
            self.dom_item_dict = dict()
            for ind in xrange(self.num_dom):
                self.dom_item_dict[ind] = set()
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rat, dom_num = (int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3]))
                if rat > 0.0:
                    rat = 1.0
                user_input.append(user)
                item_input.append(item)
                rating.append(rat)
                domain.append(dom_num)
                if train_flag:
                    self.dom_item_dict[dom_num].add(item)
                line = f.readline()
        if train_flag:
            for ind in xrange(self.num_dom):
                self.dom_item_dict[ind] = np.array(list(self.dom_item_dict[ind]))
        return np.array(user_input), np.array(item_input), np.array(rating,dtype=np.float16), np.array(domain)
        
    def load_embed_file_as_dict(self, filename):
        item_embed = dict()
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                toks = line.replace("\n","").split("::")
                itemid = int(toks[0])
                embed  = np.array(toks[1].split(" ")).astype(np.float)
                item_embed[itemid] = embed
                line = f.readline()
        return item_embed
    
    def load_file_as_dict(self, filename):
        item_embed = dict()
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                toks   = line.replace("\n","").split("::")
                keystr = toks[0].replace("(","").replace(")","").split(",")
                #print ("keystr: ",keystr)
                tup    = (int(keystr[0]),int(keystr[1]),int(keystr[2]))
                #embed  = np.array(toks[1].split(" ")).astype(np.float)
                embed  = toks[1].split(" ")
                item_embed[tup] = embed
                line = f.readline()
        return item_embed
    
    def load_rating_file_as_matrix(self, filename):        
        # Construct matrix
        mat     = sp.dok_matrix((self.num_user,self.num_item), dtype=np.float32)
        dom_mat = sp.dok_matrix((self.num_user,self.num_item), dtype=np.int16)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating, domain = (int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3]))
                if (rating > 0):
                    mat[user, item] = 1.0
                    dom_mat[user,item] = domain + 10 # to avoid explicit zero elimination prob
                line = f.readline()    
        return (mat, dom_mat)
    
    def load_rating_file_as_matrix_indiv(self, filename):        
        # Construct matrix
        mat = dict()
        for ind in xrange(self.num_dom):
            mat[ind] = sp.dok_matrix((self.num_user,self.num_item), dtype=np.float32)
            
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating, dom = (int(arr[0]), int(arr[1]), float(arr[2]), int(arr[3]))
                if (rating > 0):
                    mat[dom][user, item] = 1.0
                line = f.readline()    
        return mat
        
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_embed_as_mat(self, filename,flag='item'):
        # Construct matrix
        if flag == 'user':
            mat = np.zeros((self.num_user,self.attr_dim),dtype=np.float32)
        else:
            mat = np.zeros((self.num_itemidfiles * self.num_item,self.attr_dim),dtype=np.float32)
        ind =0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                ind = ind + 1
                toks    = line.replace("\n","").split("::")
                itemid  = int(toks[0])
                embed   = np.array(toks[1].split(" ")).astype(np.float)
                #print "embed: ",ind,len(embed), embed
                mat[itemid] = embed
                line = f.readline()
        return mat

    def get_item_count_domainwise(self, filename):
        pass
    def load_rating_file_as_matrix_train(self, filename):
        pass 

    # attention related
    def get_target_domid(self,test_file):
        return int(open(test_file).readline().split('::')[0].split(',')[2].replace(')',''))

    def load_user_items_mat(self,filename,target_flag=True):
        if target_flag == True:
            user_items_dict = self.load_user_target_items_dict(filename)
            mat = np.full((self.num_user, self.max_item_seq_length),self.num_item) # last item index is allocated for padding, numof domains are multiplied for source domain
        else:
            user_items_dict = self.load_user_source_items_dict(filename)
            mat = np.full((self.num_user,self.num_dom * self.max_item_seq_length),self.num_item) # last item index is allocated for padding, numof domains are multiplied for source domain ##decide
        for user in user_items_dict.keys():
            for item in user_items_dict[user]:
                temp_shuff_arr = np.array(user_items_dict[user])
                if target_flag == True:
                    max_seq_len = min(self.max_item_seq_length, len(temp_shuff_arr))
                else:
                    max_seq_len = min(self.max_item_seq_length * self.num_dom, len(temp_shuff_arr))
                # to shuffle inplace
                np.random.shuffle(temp_shuff_arr)
                #print max_seq_len
                #print len( mat[user,0:max_seq_len]) ,len(temp_shuff_arr[:max_seq_len])
                mat[user,0:max_seq_len] = temp_shuff_arr[:max_seq_len] # user_items_dict[user][:max_seq_len] ##shuffle selecting 20 (impt)
                #mat[user,0:len(user_items_dict[user][:max_seq_len])] = user_items_dict[user][:max_seq_len] ##shuffle selecting 20 (impt)
                #mat[user,0:len(user_items_dict[user][:self.max_item_seq_length])] = np.shuffle(user_items_dict[user])[:self.max_item_seq_length] ##shuffle selecting 20 (impt)

        return mat

    def load_user_source_items_dict(self,filename):
        return self.load_user_items_dict(filename,target_id=self.tar_dom,target_flag=False)

    def load_user_target_items_dict(self,filename):
        return self.load_user_items_dict(filename,target_id=self.tar_dom,target_flag=True)

    def load_user_items_dict(self,filename,target_id,target_flag=True):
        user_items_dict = defaultdict(list)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                toks = line.replace('\n','').split("\t")
                user, item, rating, dom = (int(toks[0]), int(toks[1]), float(toks[2]), int(toks[3]))
                if   (target_flag == True and dom == target_id):
                    user_items_dict[user].append(item)
                elif (target_flag == False and dom != target_id):
                    user_items_dict[user].append(item)
                line = f.readline()
        return user_items_dict

