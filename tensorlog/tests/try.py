import os
#import mutil
#import dataset
#
#dset = dataset.Dataset.deserialize('../datasets/cora/tmp-cache/cora-linear-train.dset')
#for (mode,x,y) in dset.minibatchIterator(batchSize=50): 
#    print str(mode),'x',mutil.pprintSummary(x),'y',mutil.pprintSummary(y)

print 'file',__file__,'dir',os.path.dirname(os.path.realpath(__file__))
print 'testdir',str(os.path.join(os.path.dirname(os.path.realpath(__file__)),"../test-data/"))
