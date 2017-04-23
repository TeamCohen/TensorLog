# default experiment and its output

expt: expt.log eval.log

expt.log: inputs/cora.cfacts
	python cora-expt.py > expt.log

eval.log: expt.log
	proppr eval tmp-cache/cora-test.examples tmp-cache/cora-test.solutions.txt --metric auc --defaultNeg > eval.log

# a quickly-viewable check on the default experiment (like a unit test)

check: actual.txt
	diff -y actual.txt expected.txt || true

unittest:
	python testexpt.py 

actual.txt: expt.log eval.log
	echo \# actual output of expt on `date` > actual.txt
	grep training.*done expt.log >> actual.txt
	grep micro eval.log >> actual.txt 

# prepare for running expt
setup: inputs/cora.cfacts

inputs/cora.cfacts: 
	cat raw/cora-*.cfacts > $@

# clean up the directory

clean:
	rm -rf *~ *.log actual.txt
	rm -rf cora-trained.db cora.db 

