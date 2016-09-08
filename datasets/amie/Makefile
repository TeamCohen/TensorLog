DATASETS:=yago2-sample
D:=yago2-sample
TAB=$(shell echo "\t")

expt: actual.txt

expt.log: $(D).expt.log
	cp $< $@



EPOCHS:=20
DEPTH:=1
PARALLEL:=30

# MEMORY currently configured for tanka:
#   ~450 GB, expressed in 1024-byte chunks
MEMORY=460800000000
%.expt.log: inputs/%.ppr inputs/%-db.cfacts inputs/%-train.exam inputs/eval.exam
	ulimit -mH $(MEMORY); \
	python amie-expt.py $* $(EPOCHS) $(DEPTH) $(PARALLEL) > $@


NAME:=$(D)-e$(EPOCHS)-d$(DEPTH)-t$(PARALLEL)
phase: $(D).expt.log
	echo actual results on `date`: $(D) > actual-$(NAME).txt
	grep 'training.*done' $< >> actual-$(NAME).txt
	grep 'eval.*on test' $< >> actual-$(NAME).txt
	cat actual-$(NAME).txt
	mkdir -p $(NAME)
	mv actual-$(NAME).txt $(NAME)
	mv tmp-cache/* $(NAME)
	mv $< $(NAME)

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

src/yago2core_facts.tsv: /remote/curtis/wcohen/data/amie/kbs/yago2/yago2core_facts.clean.notypes.tsv
	mkdir -p src
	sed 's_http://www.w3.org/2000/01/rdf-schema#__; s/$$/./' $< > $@

#inputs/%-core.cfacts: inputs
#	python bin/convert-facts.py

#inputs/%.ppr inputs/%-ruleids.cfacts: inputs
import: $(addprefix inputs/,$(D).ppr $(D)-train.exam $(D)-fortrain.cfacts)

inputs/%.ppr inputs/%-ruleids.cfacts inputs/%-core.cfacts: 
	python bin/import.py outfnstem=inputs/$* $(IMPORT)
	sort -u inputs/$*.ppr -o inputs/$*.ppr
	sort -k 2b,2 inputs/$*-core.cfacts -o inputs/$*-core.cfacts
	sort -uk 2b,2 inputs/$*-ruleids.cfacts -o inputs/$*-ruleids.cfacts
#	# TODO: Remove once issues with AMIE rules are resolved
#	cp inputs/$*.ppr.hand-tuned inputs/$*.ppr

%-db.cfacts:inputs/evalids.cfacts %-core.cfacts %-fortrain.cfacts %-ruleids.cfacts
	LC_ALL=C; cat $^ | sort -u > $@

#%-db-training.cfacts:inputs/evalids.cfacts %-fortrain.cfacts %-ruleids.cfacts
#	LC_ALL=C; cat $^ | sort -u > $@

#inputs/%.cfacts:inputs/%-core.cfacts inputs/%-ruleids.cfacts
#	cat $^ > $@

%-train.stub %-test.stub %-fortrain.cfacts: %-core.cfacts %-ruleids.cfacts
# switch to query predicates, remove queries for elided rules, and randomize
	sort -k 1b,1 $< | join -t "$(TAB)" -1 2 $(word 2,$^) - | \
	cut -f 1,3,4 | \
	sort -R --random-source=random.src > tmp.txt
# split for train/test
	N=`wc -l tmp.txt|sed 's/ .*//'`; M=`calc -p "round(0.2*$$N,0)"`; echo $$M; \
	head -n $$M tmp.txt > $*-train.stub; tail -n "+$$M" tmp.txt > $*-test.stub
# build training facts
	awk 'BEGIN{FS=OFS="\t"}{for (i=3;i<=NF;i++) {print $$1,$$2,$$i;}}' $*-test.stub > $*-fortrain.cfacts
#	awk 'BEGIN{FS=OFS="\t"}{for (i=1;i<=NF;i++) {print "entity",$$i}}' $*-train.stub >> $*-fortrain.cfacts
	grep "^inv_" $< >> $*-fortrain.cfacts
	sort -u $*-fortrain.cfacts -o $*-fortrain.cfacts
	sed -i 's/^/tr_/' $*-fortrain.cfacts

%.exam: %.stub
	sort -k 1b,1 -k 2b,2 $<  | \
	awk 'BEGIN{FS=OFS="\t"} \
	{key=$$1$$2; if (key != last) { if (last != "") { print buf; } buf = $$1 FS $$2; } \
	buf = buf FS $$3; last = key;}' | \
	sort -R --random-source=random.src | \
	sed 's/^/tri_/' > $@

inputs/%-eval.exam:inputs/%-core.cfacts inputs/eval.exam 
	join -t "$(TAB)" -1 2 -2 2 $^ | \
	awk 'BEGIN{FS=OFS="\t"}{ret = $$4 FS $$1; for (i=5; i<=NF; i++) { ret = ret FS $$i;} print ret}' > $@

inputs/eval.exam inputs/evalids.cfacts:
	python bin/convertEval.py
	sort -k 2b,2 $@ -o $@

tmp-cache:
	mkdir -p tmp-cache

#profile:
#	python -m cProfile -s cumtime supervised-expt.py r8 4

check: actual.txt
	diff -y actual.txt expected.txt || true

# TODO
actual.txt: expt.log
	echo actual results on `date`: $(D) > actual.txt
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
#import:
#	python bin/import.py yago2-sample

.SECONDARY:
