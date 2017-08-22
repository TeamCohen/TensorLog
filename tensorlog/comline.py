import getopt
import time
import logging

from tensorlog import dataset
from tensorlog import matrixdb
from tensorlog import util
from tensorlog import version

#
# utilities for reading command lines
#


def parseCommandLine(argv,extraArgConsumer=None,extraArgSpec=[],extraArgUsage=[]):

    """To be used by mains other than program to process sys.argv.  See
    the usage() subfunction for the options that are parsed.  Returns
    a dictionary mapping option names to Python objects, eg Datasets,
    Programs, ...

    If extraArgConsumer, etc are present then extra args for the
    calling program can be included after a '+' argument.
    extraArgConsumer is a label for the calling main used in help and
    error messages, and extraArgUsage is a list of strings, which will
    be printed one per line.
    """

    argspec = ["db=", "proppr", "prog=", "trainData=", "testData=", "help", "logging="]
    try:
        print "Tensorlog v%s (C) William W. Cohen and Carnegie Mellon University, 2016-2017" % version.VERSION
        optlist,args = getopt.getopt(argv, 'x', argspec)
        if extraArgConsumer:
            if args:
                if not args[0].startswith('+'): logging.warn("command-line options for %s should follow a +++ argument" % extraArgConsumer)
                extraOptList,extraArgs = getopt.getopt(args[1:], 'x', extraArgSpec)
                args = extraArgs
            else:
                extraOptList = {}
    except getopt.GetoptError:
        print 'bad option: use "--help" to get help'
        raise
    optdict = dict(optlist)
    if extraArgConsumer:
        for k,v in extraOptList:
            optdict[k] = v

    def usage():
        print 'options:'
        print ' --db file.db              - file contains a serialized MatrixDB'
        print ' --db file1.cfacts1:...    - files are parsable with MatrixDB.loadFile()'
        print ' --prog file.ppr           - file is parsable as tensorlog rules'
        print ' --trainData file.exam     - optional: file is parsable with Dataset.loadExamples'
        print ' --trainData file.dset     - optional: file is a serialized Dataset'
        print ' --testData file.exam      - optional:'
        print ' --proppr                  - if present, assume the file has proppr features with'
        print '                             every rule: {ruleid}, or {all(F): p(X,...),q(...,F)}'
        print ' --logging level           - level is warn, debug, error, or info'
        print ''
        print 'Notes: for --db, --trainData, and --testData, you are allowed to specify either a'
        print 'serialized, cached object (like \'foo.db\') or a human-readable object that can be'
        print 'serialized (like \'foo.cfacts\'). In this case you can also write \'foo.db|foo.cfacts\''
        print 'and the appropriate uncache routine will be used.'
        print '\n'.join(extraArgUsage)

    if '--logging' in optdict:
        level = optdict['--logging']
        if level=='debug':
            logging.basicConfig(level=logging.DEBUG)
        elif level=='warn':
            logging.basicConfig(level=logging.WARN)
        elif level=='error':
            logging.basicConfig(level=logging.ERROR)
        else:
            logging.basicConfig(level=logging.INFO)

    if '--help' in optdict:
        usage()
        exit(0)
    if (not '--db' in optdict) or (not '--prog' in optdict):
        usage()
        assert False,'--db and --prog are required options'

    startTime = time.time()
    def status(msg): logging.info('%s time %.3f sec mem %.3f Gb' % (msg,time.time()-startTime,util.memusage()))

    status('loading db')
    db = parseDBSpec(optdict['--db'])
    optdict['--db'] = db
    status('loading prog')
    optdict['--prog'] = parseProgSpec(optdict['--prog'],db,proppr=('--proppr' in optdict))
    status('loading prog')
    for key in ('--trainData','--testData'):
        if key in optdict:
          status('loading %s' % key[2:])
          optdict[key] = parseDatasetSpec(optdict[key],db)

    # let these be also indexed by 'train', 'prog', etc, not just '--train','--prog'
    for key,val in optdict.items():
        optdict[key[2:]] = val

    status('command line parsed')
    return optdict,args

def isUncachefromSrc(s): return s.find("|")>=0
def getCacheSrcPair(s): return s.split("|")
def makeCacheSrcPair(s1,s2): return "%s|%s" % (s1,s2)

def parseDatasetSpec(spec,db):
    """Parse a specification for a dataset, see usage() for parseCommandLine"""
    if isUncachefromSrc(spec):
        cache,src = getCacheSrcPair(spec)
        assert src.endswith(".examples") or src.endswith(".exam"), 'illegal --train or --test file'
        return dataset.Dataset.uncacheExamples(cache,db,src,proppr=src.endswith(".examples"))
    elif spec.endswith(".dset"):
        return dataset.Dataset.deserialize(spec)
    else:
        assert spec.endswith(".examples") or spec.endswith(".exam"), 'illegal --train or --test file'
        return dataset.Dataset.loadExamples(db,spec,proppr=spec.endswith(".examples"))

def parseDBSpec(spec):
    """Parse a specification for a database, see usage() for parseCommandLine"""
    if isUncachefromSrc(spec):
        cache,src = getCacheSrcPair(spec)
        result = matrixdb.MatrixDB.uncache(cache,src)
    elif spec.endswith(".db"):
        result = matrixdb.MatrixDB.deserialize(spec)
    elif spec.endswith(".cfacts"):
        result = matrixdb.MatrixDB.loadFile(spec)
    else:
        assert False,'illegal --db spec %s' %spec
    result.checkTyping()
    return result

def parseProgSpec(spec,db,proppr=False):
    """Parse a specification for a Tensorlog program,, see usage() for parseCommandLine"""
    from tensorlog import program
    return program.ProPPRProgram.loadRules(spec,db) if proppr else program.Program.loadRules(spec,db)
