# (C) William W. Cohen and Carnegie Mellon University, 2016

import sys
import re
import math
import os.path
import collections
import scipy.sparse as SS
import scipy.io as SIO
import numpy as NP
import numpy.random as NR

import mutil
import matrixdb
import declare
import logging

#
# dealing with labeled training data
#

class Dataset(object):
    
    def __init__(self,xDict,yDict):
        # dict which maps mode declaration to X matrices for training
        self.xDict = xDict
        # likewise for Y matrices
        self.yDict = yDict

    def isSinglePredicate(self):
        """Returns true if all the examples are for a single predicate."""
        return len(self.xDict.keys())==1

    def extractMode(self,mode):
        """Return a new Dataset that just contains this mode."""
        assert mode in self.xDict, 'dataset does not contain mode %s' % str(mode)
        return Dataset({mode:self.xDict[mode]}, {mode:self.yDict[mode]})

    def modesToLearn(self):
        """Return list of modes associated with the data."""
        return self.xDict.keys()

    def hasMode(self,mode):
        """True if there are examples of the mode in the dataset."""
        return (mode in self.yDict and mode in self.xDict)

    def getX(self,mode):
        """Get a matrix of all inputs for the mode."""
        return self.xDict[mode]

    def getY(self,mode):
        """Get a matrix of all desired outputs for the mode."""
        return self.yDict[mode]

    def shuffle(self):
        for mode in self.xDict:
            shuffledRowNums = NP.arange(mutil.numRows(self.xDict[mode]))
            NR.shuffle(shuffledRowNums)
            self.xDict[mode] = mutil.shuffleRows(self.xDict[mode],shuffledRowNums)
            self.yDict[mode] = mutil.shuffleRows(self.yDict[mode],shuffledRowNums)

    def minibatchIterator(self,batchSize=100,shuffleFirst=True):
        """Iterate over triples (mode,X',Y') where X' and Y' are sets of
        batchSize rows from the full data for mode, randomly selected
        (without replacement) from the dataset."""
        # randomize the order of the examples
        if shuffleFirst: self.shuffle()
        # then sample an ordering of the modes
        modeList =  self.modesToLearn()
        modeSampleDict = {}
        for modeIndex,mode in enumerate(modeList):
            numBatches = int(math.ceil( mutil.numRows(self.getX(mode)) / float(batchSize) ))
            modeSampleDict[mode] = NP.ones(numBatches,dtype='int')*modeIndex
        modeSamples = NP.concatenate(modeSampleDict.values())
        NR.shuffle(modeSamples)
        # finally produce the minibatches
        currentOffset = [0] * len(modeList)
        for modeIndex in modeSamples:
            mode = modeList[modeIndex]
            lo = currentOffset[modeIndex]
            bX = mutil.selectRows(self.getX(mode),lo,lo+batchSize)
            bY = mutil.selectRows(self.getY(mode),lo,lo+batchSize)
            currentOffset[modeIndex] += batchSize
            yield mode,bX,bY

    def pprint(self):
        return ['%s: X %s Y %s' % (str(mode),mutil.pprintSummary(self.xDict[mode]),mutil.pprintSummary(self.yDict[mode])) for mode in self.xDict]

    #
    # i/o and conversions
    # 

    def serialize(self,dir):
        """Save the dataset on disk."""
        if not os.path.exists(dir):
            os.mkdir(dir)
        dx = dict(map(lambda (k,v):(str(k),v), self.xDict.items()))
        dy = dict(map(lambda (k,v):(str(k),v), self.yDict.items()))
        SIO.savemat(os.path.join(dir,"xDict"),dx,do_compression=True)
        SIO.savemat(os.path.join(dir,"yDict"),dy,do_compression=True)
        
    @staticmethod
    def deserialize(dir):
        """Recover a saved dataset."""
        xDict = {}
        yDict = {}
        SIO.loadmat(os.path.join(dir,"xDict"),xDict)
        SIO.loadmat(os.path.join(dir,"yDict"),yDict)
        #serialization converts modes to strings so convert them
        #back.... it also converts matrices to csr
        for d in (xDict,yDict):
            for stringKey,mat in d.items():
                del d[stringKey]
                if not stringKey.startswith('__'):
                    d[declare.asMode(stringKey)] = SS.csr_matrix(mat)
        #print 'dx',xDict
        #print 'dy',yDict
        #print 'deserialized keys',xDict.keys(),yDict.keys()
        return Dataset(xDict,yDict)

    @staticmethod
    def uncacheExamples(dsetFile,db,exampleFile,proppr=True):
        """Build a dataset file from an examples file, serialize it, and
        return the de-serialized dataset.  Or if that's not necessary,
        just deserialize it.
        """
        if not os.path.exists(dsetFile) or os.path.getmtime(exampleFile)>os.path.getmtime(dsetFile):
            print 'loading exampleFile',exampleFile,'...'
            dset = Dataset.loadExamples(db,exampleFile,proppr=proppr)
            print 'serializing dsetFile',dsetFile,'...'
            dset.serialize(dsetFile)
            return dset
        else:
            print 'de-serializing dsetFile',dsetFile,'...'
            return Dataset.deserialize(dsetFile)

    @staticmethod
    def uncacheMatrix(dsetFile,db,functorToLearn,functorInDB):
        """Build a dataset file from a DB matrix as specified with loadMatrix
        and serialize it.  Or if that's not necessary, just
        deserialize it.
        """
        if not os.path.exists(dsetFile):
            print 'preparing examples from',functorToLearn,'...'
            dset = Dataset.loadMatrix(db,functorToLearn,functorInDB)
            print 'serializing dsetFile',dsetFile,'...'
            dset.serialize(dsetFile)
            return dset
        else:
            print 'de-serializing dsetFile',dsetFile,'...'
            return Dataset.deserialize(dsetFile)

    @staticmethod
    def loadMatrix(db,functorToLearn,functorInDB):
        """Convert a DB matrix containing pairs x,f(x) to training data for a
        learner.  For each row x with non-zero entries, copy that row
        to Y, and and also append a one-hot representation of x to the
        corresponding row of X.
        """
        functorToLearn = declare.asMode(functorToLearn)
        xrows = []
        yrows = []
        m = db.matEncoding[(functorInDB,2)].tocoo()
        n = db.dim()
        for i in range(len(m.data)):
            x = m.row[i]            
            xrows.append(SS.csr_matrix( ([1.0],([0],[x])), shape=(1,n) ))
            rx = m.getrow(x)
            yrows.append(rx * (1.0/rx.sum()))
        return Dataset({functorToLearn:mutil.stack(xrows)},{functorToLearn:mutil.stack(yrows)})

    @staticmethod 
    def _parseLine(line,proppr=True):
        #returns mode, x, positive y's where x and ys are symbols
        if not line.strip() or line[0]=='#':
            return None,None,None
        parts = line.strip().split("\t")
        if not proppr:
            assert len(parts)>=2, 'bad line: %r parts %r' % (line,parts)
            return declare.asMode(parts[0]+"/io"),parts[1],parts[2:]
        else:
            regex = re.compile('(\w+)\((\w+),(\w+)\)')
            mx = regex.search(parts[0])
            if not mx:
                return None,None,None
            else:
                mode = declare.asMode(mx.group(1)+"/io")
                x = mx.group(2)
                pos = []
                for ans in parts[1:]:
                    label = ans[0]
                    my = regex.search(ans[1:])
                    assert my,'problem at line '+line
                    assert my.group(1)==mode.functor,'mismatched modes %s %s at line %s' % (my.group(1),mode,line)
                    assert my.group(2)==x,'mismatched x\'s at line '+line
                    if label=='+':
                        pos.append(my.group(3))
                return mode,x,pos
        

    @staticmethod
    def loadProPPRExamples(db,fileName):
        """Convert a proppr-style foo.examples file to a two dictionaries of
        modename->matrix pairs, one for the Xs, one for the Ys"""
        loadExamples(db,fileName,proppr=True)

    @staticmethod
    def loadExamples(db,fileName,proppr=False):
        """Convert foo.exam file, where each line is of the form

          functor <TAB> x <TAB> y1 ... yk

        to two dictionaries of modename->matrix pairs, one for the Xs,
        one for the Ys.

        """
        xsTmp = collections.defaultdict(list)
        ysTmp = collections.defaultdict(list)
        regex = re.compile('(\w+)\((\w+),(\w+)\)')
        xsResult = {}
        ysResult = {}
        with open(fileName) as fp:
            for line in fp:
                pred,x,pos = Dataset._parseLine(line,proppr=proppr)
                if pred:
                    xsTmp[pred].append(x)
                    ysTmp[pred].append(pos)
            logging.info("Loading %d predicates from %s..." % (len(xsTmp),fileName))
            for pred in xsTmp.keys():
                xRows = map(lambda x:db.onehot(x), xsTmp[pred])
                xsResult[pred] = mutil.stack(xRows)
            for pred in ysTmp.keys():
                def yRow(ys):
                    accum = db.onehot(ys[0])
                    for y in ys[1:]:
                        accum = accum + db.onehot(y)
                    accum = accum * 1.0/len(ys)
                    return accum
                yRows = map(yRow, ysTmp[pred])
                ysResult[pred] = mutil.stack(yRows)
        return Dataset(xsResult,ysResult)

    #TODO refactor to also save examples in form: 'functor X Y1
    #... Yk'
    def saveProPPRExamples(self,fileName,db,append=False,mode=None):
        """Convert X and Y to ProPPR examples and store in a file."""
        fp = open(fileName,'a' if append else 'w')
        modeKeys = [mode] if mode else self.xDict.keys()
        for mode in modeKeys:
            assert mode in self.yDict
            dx = db.matrixAsSymbolDict(self.xDict[mode])
            dy = db.matrixAsSymbolDict(self.yDict[mode])
            theoryPred = mode.functor
            for i in range(max(dx.keys())):
                dix = dx[i]
                diy = dy[i]
                assert len(dix.keys())==1,'X row %d is not onehot: %r' % (i,dix)
                x = dix.keys()[0]
                fp.write('%s(%s,Y)' % (theoryPred,x))
                for y in diy.keys():
                    fp.write('\t+%s(%s,%s)' % (theoryPred,x,y))
                fp.write('\n')

if __name__ == "__main__":
    usage = 'usage: python -m dataset.py --serialize foo.cfacts|foo.db bar.exam|bar.examples glob.dset'
    if sys.argv[1]=='--serialize':
        assert len(sys.argv)==5,usage
        dbFile = sys.argv[2]
        examFile = sys.argv[3]
        dsetFile = sys.argv[4]
        if dbFile.endswith(".cfacts"):
            db = matrixdb.MatrixDB.loadFile(dbFile)
        elif dbFile.endswith(".db"):
            db = matrixdb.MatrixDB.deserialize(dbFile)
        else:
            assert False,usage
        assert examFile.endswith(".examples") or examFile.endswith(".exam"),usage
        dset = Dataset.loadExamples(db,examFile,proppr=examFile.endswith(".examples"))
        dset.serialize(dsetFile)
