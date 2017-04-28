import os
#import mutil
#import dataset
#
#dset = dataset.Dataset.deserialize('../datasets/cora/tmp-cache/cora-linear-train.dset')
#for (mode,x,y) in dset.minibatchIterator(batchSize=50): 
#    print str(mode),'x',mutil.pprintSummary(x),'y',mutil.pprintSummary(y)

class Tree(object):
    def __init__(self,left,right):
        self.left = left
        self.right = right
    def __repr__(self):
        sl = str(self.left) if self.left else "None"
        sr = str(self.right) if self.right else "None"
        return "Tree(%s, %s)" % (sl,sr)

def allTrees(k):
    if k==0: return [None]
    elif k==1: return [Tree(None,None)]
    else:
        all = []
        for j in range(0,k):
            for left in allTrees(j):
                for right in allTrees(k-j-1):
                    all.append(Tree(left,right))
        return all

def pprint(tree,depth=0):
    if tree==None:
        print '%sNone' % ('| '*depth)
    else:
        print '%sXXXX' % ('| '*depth)
        pprint(tree.left,depth+1)
        pprint(tree.right,depth+1)

def printAll(trees):
    for k,tree in enumerate(trees):
        print 'tree',k+1,tree

