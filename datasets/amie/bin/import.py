

import convertRules
import convertFacts

PREFIX_SINGLE=""
PREFIX_RECURSIVE="i_"


if __name__=="__main__":
    #convertRules.convert('/remote/curtis/wcohen/data/amie/rules/amie/amie_yago2_sample_support_2.tsv','inputs/yago2-sample',PREFIX_RECURSIVE)
    #convertRules.convert('/remote/curtis/wcohen/data/amie/rules/amie/amie_yago2_sample_support_2.tsv','inputs/yago2-sample',PREFIX_SINGLE)
    convertRules.convert('/remote/curtis/wcohen/data/amie/rules/amie/amie_yago2_sample_support_2.tsv','inputs/yago2-sample',PREFIX_SINGLE,groundInverses=False)
    #convertRules.convert('/remote/curtis/wcohen/data/amie/rules/amie/amie_yago2_sample_support_2.tsv','inputs/yago2-sample',PREFIX_RECURSIVE,groundInverses=False)
    
    #convertFacts.convert('/remote/curtis/wcohen/data/amie/kbs/yago2/yago2core.10kseedsSample.compressed.notypes.tsv','inputs/yago2-sample-core.cfacts')
    convertFacts.convert('/remote/curtis/wcohen/data/amie/kbs/yago2/yago2core.10kseedsSample.compressed.notypes.tsv','inputs/yago2-sample-core.cfacts',includeInverse=True)
