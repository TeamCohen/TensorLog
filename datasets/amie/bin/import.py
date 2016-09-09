

import convertRules
import convertFacts
import sys

#PREFIX_SINGLE=""
#PREFIX_RECURSIVE="i_"
#PREFIX_MANUAL="r"


if __name__=="__main__":
    if len(sys.argv) < 3:
        print "Usage: %s dataset name [param=value ...]" % sys.argv[0]
        sys.exit(1)
    dataset = sys.argv[1]
    name = sys.argv[2]
    opts = {}
    # defaults:
    opts['infn'] = '/remote/curtis/wcohen/data/amie/rules/amie/amie_yago2_sample_support_2.tsv'
    opts['outfnstem'] = 'inputs/%s-%s' % (dataset,name)
    opts['recursive'] = False
    
    if len(sys.argv)>3:
        for setting in sys.argv[3:]:
            kw,arg = setting.split("=")
            if kw not in opts:
                print "Syntax error: '{0}' not recognized. ".format(kw)
                print "Possible options: {0}".format(", ".join(opts.keys()))
                sys.exit(1)
            opts[kw] = arg
    
    convertRules.convert(**opts)

    #convertRules.convert(src,dst,PREFIX_RECURSIVE)
    #convertRules.convert(src,dst,PREFIX_SINGLE)
    #convertRules.convert(src,dst,PREFIX_SINGLE,groundInverses=False)
    #convertRules.convert(src,dst,PREFIX_RECURSIVE,groundInverses=False)
    #convertRules.convert(src,dst,PREFIX_MANUAL,groundInverses=False,manualRecursion=1)
    
    #convertFacts.convert('/remote/curtis/wcohen/data/amie/kbs/yago2/yago2core.10kseedsSample.compressed.notypes.tsv','inputs/yago2-sample-core.cfacts')
    convertFacts.convert('src/{0}.tsv'.format(dataset),'{0}-core.cfacts'.format(opts['outfnstem']),includeInverse=True)
