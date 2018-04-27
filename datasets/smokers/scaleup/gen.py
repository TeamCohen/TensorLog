import networkx as nx
import random as r

BATCHSIZE = 250

# generates a graph with four weakly-connected subcommunities, each a
# barabasi_albert graph, with each cluster having different labels

if __name__ == "__main__":
    seeds = [8347342984,901891212,1929282,73454129]
    for n in [100,1000,10000,100000,500000]:
        print 'generate for n',n,'...'
        with open('query-entities-%d.txt' % n,'w') as fp:
            for tag in 'a b c d'.split():
                for m in range(BATCHSIZE):
                    fp.write('%s%d\n' % (tag,r.randint(0,n-1)))
        with open('smoker-%d.cfacts' % n,'w') as fp:
            fp.write('\t'.join(['const','yes']) + '\n')
            fp.write('\t'.join(['const','no']) + '\n')
            for k in range(1,9):
                fp.write('\t'.join(['rule','r%d' % k]) + '\n')
            g1 = nx.barabasi_albert_graph(n,5,seeds[0])
            g2 = nx.barabasi_albert_graph(n,5,seeds[1])
            g3 = nx.barabasi_albert_graph(n,5,seeds[2])
            g4 = nx.barabasi_albert_graph(n,5,seeds[3])
            for tag,g in [('a',g1),('b',g2),('c',g3),('d',g4)]:
                for (i,j) in g.edges():
                    fp.write('\t'.join(['friends','%s%d' % (tag,i),'%s%d' % (tag,j)]) + '\n')
                    fp.write('\t'.join(['friends','%s%d' % (tag,j),'%s%d' % (tag,i)]) + '\n')
            # insert cross-cluster edges
            for m in range(25):
                for tag1 in 'a b c'.split():
                    for tag2 in 'a b c'.split():
                        i = r.randint(0,n)
                        j = r.randint(0,n)
                        fp.write('\t'.join(['friends','%s%d' % (tag1,i),'%s%d' % (tag2,j)]) + '\n')
                        fp.write('\t'.join(['friends','%s%d' % (tag2,j),'%s%d' % (tag1,i)]) + '\n')
            # g1 -C -S
            # g2 +C -S
            # g3 -C +S
            # g4 +C +S
            for tag,g in [('a',g1),('b',g2),('c',g3),('d',g4)]:
                for i in g.nodes():
                    fp.write('\t'.join(['person','%s%d' % (tag,i)]) + '\n')
            for tag,g in [('b',g2),('d',g4)]:
                for i in g.nodes():
                    fp.write('\t'.join(['cancer','%s%d' % (tag,i)]) + '\n')
            for tag,g in [('c',g3),('d',g4)]:
                for i in g.nodes():
                    fp.write('\t'.join(['smoker','%s%d' % (tag,i)]) + '\n')

