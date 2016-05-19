proppr compile cora
proppr set --programFiles cora.wam:cora.cfacts
time proppr ground cora-train.examples --duplicateCheck -1
time proppr train cora-train cora.params
time proppr answer cora-test.examples cora-test.solutions.txt --params cora.params --duplicateCheck -1
proppr eval cora-test.examples cora-test.solutions.txt --metric auc --defaultNeg





