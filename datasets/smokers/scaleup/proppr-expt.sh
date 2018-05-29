t=16
args="--duplicateCheck -1 --countFeatures false --prover dpr --apr eps=1e-4:alph=0.1:depth=10"
date > proppr-expt.log
echo args $args >> proppr-expt.log
#args="--duplicateCheck -1 --countFeatures false --prover ppr --apr eps=1e-4:alph=0.1:depth=10"
proppr compile smokers-for-proppr.ppr
#for n in 100 1000 10000 100000;
for n in 100 1000 10000 100000 500000; 
do
    echo executing for size $n args $args
    python queryent2proppr.py < query-entities-$n.txt > p$n.examples
    echo run proppr answer p$n.examples --programFiles smokers-for-proppr.wam:const-for-proppr.cfacts:smoker-$n.cfacts $args --threads 1 
    proppr answer p$n.examples --programFiles smokers-for-proppr.wam:const-for-proppr.cfacts:smoker-$n.cfacts $args --threads 1 > p$n.01.log
    python average-time-in-solutions.py p$n 1 < p$n.01.log >> proppr-expt.log 
#    echo run proppr answer p$n.examples --programFiles smokers-for-proppr.wam:const-for-proppr.cfacts:smoker-$n.cfacts $args --threads $t
#    proppr answer p$n.examples --programFiles smokers-for-proppr.wam:const-for-proppr.cfacts:smoker-$n.cfacts $args --threads 10 > p$n.$t.log
#    python average-time-in-solutions.py p$n $t < p$n.$t.log >> proppr-expt.log 
done
