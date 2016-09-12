DATASETS:=yago2-sample
D:=yago2-sample
TAB=$(shell echo "\t")

expt: actual.txt

expt.log: $(D).expt.log
	cp $< $@


N:=nonrecursive
EPOCHS:=20
DEPTH:=1
PARALLEL:=30
RATE:=0.1

# MEMORY currently configured for tanka:
#   ~450 GB, expressed in 1Kb chunks
MEMORY=450000000
#%.expt.log: inputs/%.ppr inputs/%-db.cfacts inputs/%-train.exam inputs/eval.exam
%.expt.log: inputs/%.ppr inputs/%-db.cfacts inputs/%-train.exam inputs/%-eval.exam
	ulimit -m $(MEMORY); \
	python amie-expt.py $* $(EPOCHS) $(DEPTH) $(PARALLEL) $(RATE)> $@


NAME:=$(D)-$(N)-e$(EPOCHS)-d$(DEPTH)-t$(PARALLEL)
STEM:=$(D)-$(N)
phase: $(STEM).expt.log
	echo actual results on `date`: $(STEM) depth $(DEPTH) > actual-$(NAME).txt
	grep 'training.*done' $< >> actual-$(NAME).txt
	grep 'eval.*on test' $< >> actual-$(NAME).txt
	cat actual-$(NAME).txt
	rm -rf $(NAME)
	mkdir $(NAME)
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

proppr:
	proppr set --programFiles inputs/$(STEM).wam:inputs/$(STEM)-db.cfacts --duplicateCheck -1 --threads $(PARALLEL)
	proppr answer $(NAME)/$(STEM)-train.examples $(NAME)/proppr.$(STEM)-train.solutions.txt | tee $(NAME)/proppr.log
	proppr answer $(NAME)/$(STEM)-eval.examples $(NAME)/proppr.$(STEM)-eval.solutions.txt | tee -a $(NAME)/proppr.log
	proppr eval $(NAME)/$(STEM)-eval.examples --metric auc --defaultNeg | tee -a $(NAME)/proppr.log

setup: inputs tmp-cache src

inputs:
	mkdir -p inputs

src:
	mkdir -p src

src/yago2-core.tsv: /remote/curtis/wcohen/data/amie/kbs/yago2/yago2core_facts.clean.notypes.tsv src
	sed 's_http://www.w3.org/2000/01/rdf-schema#__; s/$$/./' $< > $@

src/yago2-sample.tsv: /remote/curtis/wcohen/data/amie/kbs/yago2/yago2core.10kseedsSample.compressed.notypes.tsv src
	ln -s $< $@

#inputs/%-core.cfacts: inputs
#	python bin/convert-facts.py

#inputs/%.ppr inputs/%-ruleids.cfacts: inputs
#import: $(addprefix inputs/,$(D).ppr $(D)-train.exam $(D)-fortrain.cfacts)

inputs/$(D)-$(N).ppr inputs/$(D)-$(N)-ruleids.cfacts inputs/$(D)-$(N)-core.cfacts: src/$(D).tsv
	python bin/import.py $(D) $(N) $(IMPORT)
	sort -u inputs/$(D)-$(N).ppr -o inputs/$(D)-$(N).ppr
	sort -k 2b,2 inputs/$(D)-$(N)-core.cfacts -o inputs/$(D)-$(N)-core.cfacts
	sort -uk 2b,2 inputs/$(D)-$(N)-ruleids.cfacts -o inputs/$(D)-$(N)-ruleids.cfacts

#%-db.cfacts:inputs/evalids.cfacts %-core.cfacts %-fortrain.cfacts %-ruleids.cfacts
%-db.cfacts: %-foreval.cfacts %-fortrain.cfacts %-ruleids.cfacts
	LC_ALL=C; cat $^ | sort -u > $@

#%-db-training.cfacts:inputs/evalids.cfacts %-fortrain.cfacts %-ruleids.cfacts
#	LC_ALL=C; cat $^ | sort -u > $@

#inputs/%.cfacts:inputs/%-core.cfacts inputs/%-ruleids.cfacts
#	cat $^ > $@

SET_M:=M=5000;
# SET_M:=N=`wc -l tmp.txt|sed 's/ .*//'`; M=`calc -p "round(0.2*$$N,0)"`;

%-queryable.cfacts: %-core.cfacts %-ruleids.cfacts
# remove queries for elided rules, and randomize
	sort -k 1b,1 $< | join -t "$(TAB)" -1 2 $(word 2,$^) - | \
	cut -f 1,3,4 > $@
	sort -R --random-source=random.src $@ -o $@


%-train.stub %-test.stub %-fortrain.cfacts: %-queryable.cfacts %-core.cfacts
# split for train/test
	$(SET_M) echo $$M; \
	head -n $$M $< > $*-train.stub; \
	tail -n "+$$M" $< > $*-test.stub
# build training facts
	awk 'BEGIN{FS=OFS="\t"}{for (i=3;i<=NF;i++) {print $$1,$$2,$$i;}}' $*-test.stub > $*-fortrain.cfacts ; \
	awk 'BEGIN{FS=OFS="\t"}{for (i=2;i<=NF;i++) {print "entity",$$i;}}' $*-train.stub | sort -u >> $*-fortrain.cfacts ; \
	grep "^inv_" $(word 2,$^) >> $*-fortrain.cfacts ; \
	sort -u $*-fortrain.cfacts -o $*-fortrain.cfacts ; \
	sed -i 's/^/tr_/' $*-fortrain.cfacts

NUM_EVAL:=100
%-eval.stub %-foreval.cfacts: %-test.stub
	sed 's/\t/:/' $< | sort -k 1b,1 > tmp.txt
	sed 's/\t.*//' tmp.txt | \
	sort -u | sort -R --random-source=random.src | \
	head -$(NUM_EVAL) | sort -k 1b,1 > keys.txt
	join -t "$(TAB)" keys.txt tmp.txt | \
	sed 's/:/\t/' > $*-eval.stub
	cut -f 2 $*-eval.stub | sort -u | sed 's/^/entity\t/' > $*-foreval.cfacts
	join -t "$(TAB)" -v 2 keys.txt tmp.txt | \
	sed 's/:/\t/' >> $*-foreval.cfacts
	rm tmp.txt keys.txt

%.exam: %.stub
	sort -k 1b,1 -k 2b,2 $<  | \
	awk 'BEGIN{FS=OFS="\t"} \
	{key=$$1$$2; if (key != last) { if (last != "") { print buf; } buf = $$1 FS $$2; } \
	buf = buf FS $$3; last = key;} END{print buf;}' | \
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
