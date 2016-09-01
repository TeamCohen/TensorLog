# Convert facts from AMIE format to TensorLog.
# /remote/curtis/wcohen/data/amie/kbs/yago2$ head yago2core.10kseedsSample.compressed.notypes.tsv
# Sample input:
"""
<Azerbaijan>    <hasCapital>    <Baku>.
<Azerbaijan>    <dealsWith>     <People's_Republic_of_China>.
<Azerbaijan>    <dealsWith>     <Germany>.
<Azerbaijan>    <dealsWith>     <Georgia_(country)>.
<Azerbaijan>    <dealsWith>     <Italy>.
<Azerbaijan>    <dealsWith>     <Indonesia>.
<Azerbaijan>    <dealsWith>     <Iran>.
<Azerbaijan>    <dealsWith>     <Israel>.
<Azerbaijan>    <dealsWith>     <Japan>.
<Azerbaijan>    <dealsWith>     <Russia>.
"""
# Sample output:
"""
hasCapital    s_Azerbaijan    s_Baku
dealsWith     s_Azerbaijan    s_Peoples_Republic_of_China
dealsWith     s_Azerbaijan    s_Germany
dealsWith     s_Azerbaijan    s_Georgia_country
...
"""

import sys
import re

STOP=re.compile("['(),]")

def convert(ifn, ofn, includeInverse=False):
    def sanitize(s):
        return 's_%s' % STOP.sub("",s[1:-1])
    with open(ifn,'r') as f, open(ofn,'w') as o:
        for line in f:
            line = line.strip()
            # drop trailing '.'
            parts = line[:-1].split("\t")
            pred = [parts[1][1:-1],sanitize(parts[0]),sanitize(parts[2])]
            o.write("\t".join(pred))
            o.write("\n")
            if includeInverse:
                o.write("\t".join(["inv_"+pred[0],pred[2],pred[1]]))
                o.write("\n")

if __name__ == '__main__':
    convert('/remote/curtis/wcohen/data/amie/kbs/yago2/yago2core.10kseedsSample.compressed.notypes.tsv','inputs/yago2-sample-core.cfacts')
