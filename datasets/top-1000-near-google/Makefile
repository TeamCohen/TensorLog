# default experiment and its output

expt: expt.log eval.log

expt.log:
	python top-1000-near-google.py > expt.log

eval.log: expt.log
	proppr eval top-1000-near-google-test.examples top-1000-near-google-test.solutions.txt --metric auc --defaultNeg > eval.log

# a quickly-viewable check on the default experiment (like a unit test)

check: actual.txt
	diff -y actual.txt expected.txt || true

actual.txt: expt.log eval.log
	echo \# actual output of expt on `date` > actual.txt
	grep training.*done expt.log >> actual.txt
	grep micro eval.log >> actual.txt 

# prepare for running expt
setup: 
	mkdir -p tmp-cache

# clean up the directory

clean:
	rm -rf *~ *.log actual.txt
	rm -rf *.solutions.txt *.examples
