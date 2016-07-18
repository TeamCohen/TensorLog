expt: actual.txt

actual.txt:
	echo \# actual result on `date` > actual.txt
	python smokers-expt.py >> actual.txt

check:
	paste actual.txt expected.txt

clean:
	rm -f *~ *.pyc actual.txt

