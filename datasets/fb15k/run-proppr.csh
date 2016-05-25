echo running for $1

proppr compile fb-learned
proppr set --programFiles fb-learned.wam:fb.cfacts
time proppr ground $1-train.examples --duplicateCheck -1
time proppr train $1-train $1.params
time proppr answer $1-test.examples proppr-test.solutions.txt --params $1.params --duplicateCheck -1
proppr eval $1-test.examples proppr-test.solutions.txt --metric auc --defaultNeg





