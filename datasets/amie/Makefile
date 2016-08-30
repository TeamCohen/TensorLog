DATASETS:=yago2-sample
TAB=$(shell echo "\t")

expt: expt.log

expt.log: yago2-sample.expt.log
	cp $< $@

%.expt.log: inputs/%.ppr inputs/%.cfacts
	python amie-expt.py $* > $@

#TODO
eval:
	python -m expt \
	    --db 'inputs/$*.db|inputs/$*-corpus.cfacts' --prog inputs/$*.ppr --proppr \
	    --train 'inputs/$*-train.dset|inputs/$*-train.exam' \
	    --test 'inputs/$*-test.dset|inputs/$*-test.exam' \
	    +++ \
	    --learner plearn.ParallelFixedRateGDLearner --learnerOpts "\"{'epochs':60,'parallel':55,'rate':20}\"" \
	    --savedModel tmp-cache/$*.model

setup: inputs tmp-cache $(addprefix inputs/,$(addsuffix -train.exam,$(DATASETS)) $(addsuffix -test.exam,$(DATASETS)))

inputs:
	mkdir -p inputs
#	python bin/import.py yago2-sample

inputs/%-core.cfacts: inputs
	python bin/convert-facts.py

inputs/%.ppr inputs/%-ruleids.cfacts: inputs
	python bin/convert-rules.py
	sort -uk 2b,2 inputs/$*-ruleids.cfacts -o inputs/$*-ruleids.cfacts
	# TODO: Remove once issues with AMIE rules are resolved
	cp inputs/$*.ppr.hand-tuned inputs/$*.ppr

inputs/%.cfacts:inputs/%-core.cfacts inputs/%-ruleids.cfacts
	cat $^ > $@

%-train.stub %-test.stub: %-core.cfacts %-ruleids.cfacts
# switch to query predicates, remove queries for elided rules, and randomize
	sed 's/^/i_/' $< | sort -k 1b,1 | join -t "$(TAB)" -1 2 $(word 2,$^) - | \
	cut -f 1,3,4 | \
	sort -R --random-source=random.src > tmp.txt
# split for train/test
	N=`wc -l tmp.txt|sed 's/ .*//'`; M=`calc -p "round(0.7*$$N,0)"`; echo $$M; \
	head -n $$M tmp.txt > $*-train.stub; tail -n "+$$M" tmp.txt > $*-test.stub

%.exam: %.stub
	sort -k 1b,1 -k 2b,2 $<  | \
	awk 'BEGIN{FS=OFS="\t"} \
	{key=$$1$$2; if (key != last) { if (last != "") { print buf; } buf = $$1 FS $$2; } \
	buf = buf FS $$3; last = key;}' | \
	sort -R --random-source=random.src > $@

tmp-cache:
	mkdir -p tmp-cache

#profile:
#	python -m cProfile -s cumtime supervised-expt.py r8 4

check: actual.txt
	diff -y actual.txt expected.txt || true

# TODO
actual.txt: expt.log
	echo actual results on `date`: yago2-sample > actual.txt
	grep 'training.*done' expt.log >> actual.txt
	grep 'eval.*on test' expt.log >> actual.txt

clean:
	rm -rf tmp-cache/* *~ bin/*~ expt.log

reset:
	rm -rf inputs/yago2-sample*.d*

#
# import the original inputs, by default from /remote/curtis/wcohen/data/amie
#

#TODO
import:
	python bin/import.py yago2-sample

.SECONDARY:
