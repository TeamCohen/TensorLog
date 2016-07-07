test: setup
	python fb-expt.py

setup: fb.cfacts train.exam valid.exam

learned-rules.ppr ruleids.cfacts: train.cfacts
	python bin/convert-rules.py train.cfacts inputs/learned-rules.ppr learned-rules.ppr ruleids.cfacts

fb.cfacts: ruleids.cfacts train.cfacts
	cat ruleids.cfacts train.cfacts > fb.cfacts

train.cfacts train.exam:
	python bin/partition.py inputs/train.clean 80 tmp-cache/train4db.triples tmp-cache/train4exam.triples
	python bin/triples2cfacts.py tmp-cache/train4db.triples > train.cfacts
	python bin/triples2exam.py train.cfacts tmp-cache/train4exam.triples > train.exam 

valid.exam: train.cfacts
	python bin/triples2exam.py train.cfacts inputs/valid.clean > valid.exam 

clean:
	rm -f *~ train.cfacts train.exam valid.exam fb.cfacts learned-rules.ppr ruleids.cfacts 
	rm -rf train.dset valid.dset fb.db
