import sys
import shutil
import os.path

def loadList(fromFile):
    return [line.strip() for line in open(fromFile)]

def loadTuples(fromFile):
    return [line.strip().split("\t") for line in open(fromFile)]

def genTextCatPPR(toFile,labels):
    fp = open(toFile,'w')
    for y in labels:
        fp.write('predict(X,Y) :- assign(Y,%s) {all(F): hasWord(X,W),%sPair(W,F)}.\n' % (y,y))

def genExamples(toFile,examples):
    fp = open(toFile,'w')    
    for (x,y) in examples:
        x = fixedToken(x,'d')
        fp.write('\t'.join(['predict',x,y]) + '\n')

def fixedToken(x,prefix):
    return prefix+x if "0" <= x[0] <= "9" else x

def genCorpusFacts(toFile,labels,tuples):
    fp = open(toFile,'w')
    # triples might or might not be weighted, which complicates things
    def fixedTuples(tuples):
        for tup in tuples:
            docid = fixedToken(tup[1],'d')
            yield [tup[0],docid] + tup[2:]

    for tup in fixedTuples(tuples):    
        # write the hasWord fact
        fp.write('\t'.join(tup) + '\n')
    for tup in fixedTuples(tuples):    
        # write the yyyPair(www, www_yyy) facts
        for y in labels:
            word = tup[2]
            rel = '%sPair' % y
            pair = '%s_%s' % (word,y)
            fp.write('\t'.join([rel,word,pair]) + '\n')
    for tup in fixedTuples(tuples):    
        # write the weighted(www_yyy) facts
        for y in labels:
            word = tup[2]
            pair = '%s_%s' % (word,y)
            fp.write('\t'.join(['weighted',pair]) + '\n')
    for y in labels:
        # write the label(yyy) facts
        fp.write('\t'.join(['label',y]) + '\n')

if __name__ == "__main__":
    # usage: python import.py stem [srcdir]
    stem = sys.argv[1]
    srcdir = '/afs/cs.cmu.edu/user/wcohen/proppr-1/textcat/%s-inputs/' % stem
    if len(sys.argv)>2:
        srcdir = sys.argv[2]
    dstdir = 'inputs'

    print 'generate theory...'
    labels = loadList(os.path.join(srcdir,'labels.txt'))
    genTextCatPPR(os.path.join(dstdir,'%s-textcat.ppr' % stem), labels)

    print 'generate corpus.cfacts...'
    corpusTriples = loadTuples(os.path.join(srcdir,'corpus.graph'))
    # generate the corpus file supervised-learning theory
    genCorpusFacts(os.path.join(dstdir,'%s-corpus.cfacts' % stem), labels, corpusTriples)

    print 'generate train.exam...'
    trainExamples = loadTuples(os.path.join(srcdir,'labeled.txt'))
    genExamples(os.path.join(dstdir,'%s-train.exam' % stem), trainExamples)

    print 'generate test.exam...'
    testExamples = loadTuples(os.path.join(srcdir,'test-examples.txt'))
    genExamples(os.path.join(dstdir,'%s-test.exam' % stem), testExamples)

