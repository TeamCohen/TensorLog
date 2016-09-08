
"""
<Eva_Bartok>    <isCitizenOf>   <United_Kingdom>        0.8011806416    0.71875 ManualEvaluation        FALSE
"""


from amie import *


def convert(infn,outfnstem):
    db = {}
    with open(infn,'r') as f, open(outfnstem+"ids.cfacts",'w') as o:
        for line in f:
            # skip partition definition lines and blank lines
            if line.find("ManualEvaluation") <0: continue
            # skip negative labels?
            if line.find("FALSE") >0: continue
            
            line = line.strip()
            parts = line.split("\t")
            pred = [sanitize(x) for x in [parts[1],parts[0],parts[2]]]
            pred[0] = 'i'+pred[0][1:]
            key = tuple(pred[:2])
            if key not in db: db[key] = []
            db[key].append(pred[-1])
            o.write("entity\t{0}\t{1}\n".format(pred[1],pred[2]))
    with open(outfnstem+".exam",'w') as o:
        for k,v in db.items():
            o.write("\t".join(k))
            o.write("\t")
            o.write("\t".join(v))
            o.write("\n")

if __name__=='__main__':
    convert('/remote/curtis/wcohen/data/amie/evals/joint-prediction.tsv','inputs/eval')
