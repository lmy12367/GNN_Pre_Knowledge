import networkx as nx
print(nx.__version__)

G=nx.Graph()

G.add_nodes_from(['v1','v2','v3','v4','v5'])
G.add_edges_from([('v1','v2'),('v1','v3'),('v2','v3'),('v2','v4'),('v3','v4'),('v4','v5')])
print(nx.is_directed(G))
print(nx.is_weighted(G))

G1=nx.draw(G,with_labels=True,pos=nx.spring_layout(G))







