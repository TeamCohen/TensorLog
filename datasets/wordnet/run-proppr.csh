proppr compile wnet-learned
proppr settings --programFiles wnet-learned.wam:wnet.cfacts
time proppr ground hypernym-train.examples --duplicateCheck -1
time proppr train hypernym-train hypernym.params
time proppr answer hypernym-test.examples proppr-test.solutions.txt --params hypernym.params --duplicateCheck -1
proppr eval hypernym-test.examples proppr-test.solutions.txt --metric auc --defaultNeg





