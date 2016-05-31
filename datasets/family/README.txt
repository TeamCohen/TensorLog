taken from regression benchmark at /afs/cs/project/proppr-1/regression/structureLearning/kinship


RULES

kinship-train-isg.ppr from 5/30/2016 regression run at duck:~krivard/ProPPR-nighlies/check/structureLearning

learnedPred( and rel( were lowered to their predicate argument, and anonymous predicate rules were removed.

rule features {} had their arguments merged using a $ delimiter.


FACTS

kinship-train.cfacts and kinship-test.cfacts were combined and duplicates removed, then lowered.

kinship-rules.cfacts was generated from the rule features in the .ppr file, and the "rule" facts added for each feature predicate lr_if, lr_ifInv, lr_chain


EXAMPLES

interp( was lowered to its predicate argument

examples with no + labels were removed


