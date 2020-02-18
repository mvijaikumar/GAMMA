import numpy as np
import scipy.sparse as sp
from time import time
from Dataset import Dataset
import subprocess as sub
import pandas as pd
import sys,pdb

class ItemView_Dataset(Dataset):
    def __init__(self,args):
        Dataset.__init__(self,args)

        self.same_entity_list       = eval(args.same_entity)
        assert len(self.same_entity_list) == args.num_views, 'length of same_entity and num_views should be same.'
        
        self.item_view_matrix,self.num_row,self.num_col,self.adjacency_view_matrix = [],[],[],[]
        for view in range(args.num_views):
            num_row, num_col = self.get_row_column_count(self.embed_path + ".view_matrix" + str(view+1))
            self.num_row.append(num_row) # this is only for test purpose. self.num_item is used instead
            self.num_col.append(num_col)

            if self.same_entity_list[view] == -1: #-1 means entities are different
                self.item_view_matrix.append(self.load_rating_file_as_matrix_for_views(self.embed_path + ".view_matrix" + str(view+1),self.num_row[view],self.num_col[view]))
                self.adjacency_view_matrix.append(self.get_adjacency_matrix_sparse(self.item_view_matrix[view],self.item_view_matrix[view].T)) # Note num_items is used
            else: # 1 or other number means it is same entity both side
                self.item_view_matrix.append(self.load_rating_file_as_matrix_for_views(self.embed_path + ".view_matrix" + str(view+1), max(self.num_row[view],self.num_col[view]),max(self.num_row[view],self.num_col[view])))
                _A_obs = self.item_view_matrix[view] + self.item_view_matrix[view].T # Note num_items is used
                _A_obs[_A_obs > 1] = 1
                self.adjacency_view_matrix.append(_A_obs) # Note num_items is used

    def get_row_column_count(self, filename):
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline().strip()
            while line != None and line != "":
                arr         = line.split("\t")
                u, i        = int(arr[0]), int(arr[1])
                num_users   = max(num_users, u)
                num_items   = max(num_items, i)
                line = f.readline()
        return num_users+1, num_items+1

    def load_rating_file_as_matrix_for_views(self, filename, num_row, num_col):        
        mat     = sp.dok_matrix((num_row,num_col), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = (int(arr[0]), int(arr[1]))
                mat[user, item] = 1.0
                line = f.readline()    
        return mat

    def get_adjacency_matrix_sparse(self,mat1,mat2): ## exactly same as param
        num_row,num_col = (mat1.shape[0] + mat2.shape[0], mat1.shape[1] + mat2.shape[1])
        mat = sp.lil_matrix((num_row,num_col),dtype=np.float32)
        assert num_row == num_col, 'In adj matrix conv. row and col should be equal.'
        mat[0:mat1.shape[0],mat1.shape[0]:] = mat1.astype(np.float32).tolil()
        mat[mat1.shape[0]:,0:mat1.shape[0]] = mat2.astype(np.float32).tolil()
        return mat.tocsr()

    def get_num_social(self,fname):
        df = pd.read_csv(fname,delimiter='\t',header=None)
        return df[1].max()+1

    def load_item_embed_as_mat(self, filename, zero_flag=False):
        # Construct matrix
        mat = np.zeros((self.num_item,self.item_attr_dim),dtype=np.float32)
        item_set = set()
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                toks    = line.replace("\n","").split("::")
                itemid  = int(toks[0])
                embed   = np.array(toks[1].split(" ")).astype(np.float)
                mat[itemid] = embed
                line = f.readline()
                item_set.add(itemid)
            avg = np.mean(mat,axis=0)
            if zero_flag == False:
                for itemid in range(self.num_item):
                    if itemid not in item_set:
                        mat[itemid] = avg
        return mat

