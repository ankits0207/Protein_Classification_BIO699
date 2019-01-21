import pickle
import networkx as nx

mesoPkl = 'MesoPkl'
thermoPkl = 'ThermoPkl'


class MyStructure:
    def __init__(self, pdbId, edgeList):
        self.pdbId = pdbId
        self.edgeList = edgeList


class Edge:
    def __init__(self, res1, res2, dist):
        self.res1 = res1
        self.res2 = res2
        self.dist = dist


def Convert(MyNetworks):
    listOfNxGraphs = []
    idx = 1
    for network in MyNetworks:
        print(str(idx) + ':' + str(len(MyNetworks)))
        G = nx.Graph(name=network.pdbId)
        el = network.edgeList
        for e in el:
            G.add_edge(str(e.res1), str(e.res2), weight=e.dist)
        if G.number_of_edges() > 0:
            listOfNxGraphs.append(G)
        idx += 1
    return listOfNxGraphs


with open(mesoPkl, 'rb') as f:
    MesoList = pickle.load(f)
mesoNxGraphs = Convert(MesoList)
with open(mesoPkl + 'Nx', 'wb') as f:
    pickle.dump(mesoNxGraphs, f)

with open(thermoPkl, 'rb') as f:
    ThermoList = pickle.load(f)
thermoNxGraphs = Convert(ThermoList)
with open(thermoPkl + 'Nx', 'wb') as f:
    pickle.dump(thermoNxGraphs, f)
print('Done')
