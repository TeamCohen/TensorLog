default:
	echo make what?

fb.db: fb.cfacts
	time python $(TLOG)/matrixdb.py --serialize fb.cfacts fb.db

fb.cfacts: fb-ruleids.cfacts
	cut -f2- raw/train.cfacts raw/valid.cfacts > fb-core.cfacts
	python bin/convert-examples.py
	cat fb-*.cfacts | sort | uniq > fb.cfacts

fb-learned.ppr fb-ruleids.cfacts:
	python bin/convert-rules.py

fb-learned.preds: fb-learned.ppr
	sed 's/[(].*//' $< | sort -k 1b,1 | uniq > $@

TAB=$(shell echo "\t")
inputs/%.examples: raw/%.examples fb-learned.preds
	awk 'BEGIN{FS=OFS="\t"}{for (i=1;i<NF;i++) { sub("interp[(]","",$$i);sub(",","(",$$i);} foo=$$1; sub("[(].*","",foo); print foo,$$0}' $< | \
	sort -k 1b,1 | \
	join -t "${TAB}" - fb-learned.preds | \
	sed 's/[^\t]*\t//' > $@

setup: inputs/train.examples inputs/valid.examples

proppr-version:
	mkdir -p proppr
	proppr compile fb-learned.ppr
	proppr set --programFiles fb-learned.wam:fb.cfacts
	proppr ground inputs/train.examples proppr/train.grounded
	proppr train proppr/train.grounded --epochs 10
	proppr answer inputs/valid.examples proppr/all-test.solutions.txt
	proppr eval inputs/valid.examples proppr/all-test.solutions.txt --metric auc --defaultNeg


clean:
	rm -f *.pyc *.prof *~
