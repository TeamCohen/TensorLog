# 
# 

expt.log: inputs/fb.cfacts inputs/train.exam inputs/valid.exam
	python fb-expt.py 10 20 i_people_person_nationality > expt.log

check: actual.txt
	diff -y actual.txt expected.txt || true

actual.txt: expt.log
	echo \# actual output of expt on `date` > actual.txt
	grep training.*done expt.log >> actual.txt
	grep eval.*\.trained.*train expt.log >> actual.txt 

# prepare for an experiment

setup: inputs/fb.cfacts inputs/train.exam inputs/valid.exam

inputs/learned-rules.ppr inputs/ruleids.cfacts: inputs/train.cfacts
	python bin/convert-rules.py inputs/train.cfacts raw/learned-rules.ppr inputs/learned-rules.ppr inputs/ruleids.cfacts

inputs/fb.cfacts: inputs/ruleids.cfacts inputs/train.cfacts
	cat inputs/ruleids.cfacts inputs/train.cfacts > inputs/fb.cfacts

inputs/train.cfacts inputs/train.exam:
	python bin/partition.py raw/train.clean 80 tmp-cache/train4db.triples tmp-cache/train4exam.triples
	python bin/triples2cfacts.py tmp-cache/train4db.triples > inputs/train.cfacts
	python bin/triples2exam.py inputs/train.cfacts tmp-cache/train4exam.triples > inputs/train.exam 

inputs/valid.exam: inputs/train.cfacts
	python bin/triples2exam.py inputs/train.cfacts raw/valid.clean > inputs/valid.exam 

# clean up

clean:
	rm -f *~ actual.txt expt.log

