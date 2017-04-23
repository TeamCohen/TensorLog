# default experiment and its output

expt: expt.log eval.log

# expt.py reads inputs/* and outputs a log file
expt.log: inputs inputs/wnet.cfacts inputs/wnet-learned.ppr inputs/wnet-train.exam inputs/wnet-valid.exam
	python expt.py > expt.log

eval.log: expt.log
	proppr eval tmp-cache/wnet-test.examples tmp-cache/hypernym-test.solutions.txt --metric auc --defaultNeg > eval.log

# prepare for running expt

setup: inputs tmp-cache inputs/wnet.cfacts inputs/wnet-learned.ppr inputs/wnet-train.exam inputs/wnet-valid.exam

inputs:
	mkdir $@

tmp-cache:
	mkdir $@

# convert raw/train.exam, raw/valid.exam to inputs/train.exam, ...
inputs/wnet-train.exam inputs/wnet-valid.exam:
	python bin/convert-examples.py

# convert the raw/(train,valid).cfacts and the inputs/wnet-ruleids.cfacts and generate wnet.cfacts
inputs/wnet.cfacts: inputs/wnet-ruleids.cfacts
	cut -f2- raw/train.cfacts raw/valid.cfacts | perl -nae 'print join("\t",$$F[0],"s".$$F[1],"s".$$F[2]),"\n"' > inputs/wnet-core.cfacts
	cat inputs/wnet-*.cfacts > inputs/wnet.cfacts

# convert the rules from raw/train-learned.ppr to inputs/wnet-learned.ppr and inputs/wnet-ruleids.cfacts
inputs/wnet-learned.ppr inputs/wnet-ruleids.cfacts:
	python bin/convert-rules.py

# a quickly-viewable check on the default experiment (like a unit test)

check: actual.txt
	diff -y actual.txt expected.txt || true

actual.txt: expt.log eval.log
	echo \# actual output of expt on `date` > actual.txt
	grep training.*done expt.log >> actual.txt
	grep micro eval.log >> actual.txt 

# clean up the directory

clean:
	rm -f *.pyc *.prof *~ expt.log

