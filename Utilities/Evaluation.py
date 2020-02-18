import math,heapq,multiprocessing,sys
import numpy as np
from time import time

_posItemlst   = None
_itemMatrix   = None
_predMatrix   = None
_k            = None
_matShape     = None

def evaluate_model(posItemlst, itemMatrix, predMatrix, k, num_thread):
    
    global _posItemlst
    global _itemMatrix
    global _predMatrix
    global _k
    global _matShape
    
    _posItemlst = posItemlst
    _itemMatrix = itemMatrix
    _predMatrix = predMatrix
    _k          = k    
    _matShape   = itemMatrix.shape
    
    num_inst    = _matShape[0]
    #===============================================================
    
    #print "Inside eval vijai: "
    #print _posItemlst, _itemMatrix, _predMatrix 
    
    hits, ndcgs = [],[]
    if(num_thread > 1):    
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating,range(num_inst))
        
        pool.close()
        pool.join()
        hits  = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    
    # Single thread        
    for ind in xrange(num_inst):
        (hr,ndcg) = eval_one_rating(ind)        
        hits.append(hr)
        ndcgs.append(ndcg)
    #print ("hits and ndcgs: ",hits,ndcgs)
    return (hits, ndcgs)

def eval_one_rating(ind):
    map_item_score = {}
    predictions    = _predMatrix[ind]
    items          = _itemMatrix[ind]
    gtItem         = _posItemlst[ind]
    
    for i in xrange(_matShape[1]): ## parallelaize by assigning array to array in dict
        item = items[i]
        map_item_score[item] = predictions[i]
    # Evaluate top rank list
    ranklist = heapq.nlargest(_k, map_item_score, key=map_item_score.get)
    #ranklist = heapq.nsmallest(_k, map_item_score, key=map_item_score.get)
    hr       = getHitRatio(ranklist, gtItem)
    ndcg     = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
