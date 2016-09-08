

import convertRules
import convertFacts
import sys

PREFIX_SINGLE=""
PREFIX_RECURSIVE="i_"
PREFIX_MANUAL="r"


if __name__=="__main__":
    opts = {}
    # defaults:
    opts['infn'] = '/remote/curtis/wcohen/data/amie/rules/amie/amie_yago2_sample_support_2.tsv'
    opts['outfnstem'] = 'inputs/yago2-sample'
    opts['recursive'] = False
    
    if len(sys.argv)>1:
        for setting in sys.argv[1:]:
            kw,arg = setting.split("=")
            if kw not in opts and kw is not 'manualRecursion':
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
    convertFacts.convert('/remote/curtis/wcohen/data/amie/kbs/yago2/yago2core.10kseedsSample.compressed.notypes.tsv','{0}-core.cfacts'.format(opts['outfnstem']),includeInverse=True)
