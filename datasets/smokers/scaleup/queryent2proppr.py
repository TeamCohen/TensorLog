import sys

if __name__ == "__main__":
    xs = []
    for line in sys.stdin:
        xs.append(line.strip())
    for rel in ["t_stress", "t_influences","t_cancer_spont", "t_cancer_smoke"]:
        for x in xs:
            print '%s(%s,Y)' % (rel,x)

        
        
        
