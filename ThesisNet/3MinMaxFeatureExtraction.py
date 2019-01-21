import pickle
import networkx as nx

mesoFile = 'Meso'
mesoNxPkl = mesoFile + 'PklNx'
with open(mesoNxPkl, 'rb') as f:
    mesoList = pickle.load(f)

thermoFile = 'Thermo'
thermoNxPkl = thermoFile + 'PklNx'
with open(thermoNxPkl, 'rb') as f:
    thermoList = pickle.load(f)

degList = []
clustCoeffList = []
closenessCentralityList = []
degreeCentralityList = []
edgeBetweenessCentralityList = []
cfcCentralityList = []
sgCentralityList = []


def populateLists(myList):
    idx = 0
    for elt in myList:
        if nx.is_connected(elt):
            print(str(idx) + ':' + str(len(myList)) + '-' + 'Degree')
            degrees = elt.degree()
            for degree in degrees:
                degList.append(degree[1])
            print(str(idx) + ':' + str(len(myList)) + '-' + 'Clust Coeff')
            clusteringCoeffs = nx.clustering(elt)
            for k, v in clusteringCoeffs.items():
                clustCoeffList.append(v)
            print(str(idx) + ':' + str(len(myList)) + '-' + 'Closeness Cen')
            closenessCentralities = nx.closeness_centrality(elt)
            for k, v in closenessCentralities.items():
                closenessCentralityList.append(v)
            print(str(idx) + ':' + str(len(myList)) + '-' + 'Degree Cen')
            degreeCentralities = nx.degree_centrality(elt)
            for k, v in degreeCentralities.items():
                degreeCentralityList.append(v)
            print(str(idx) + ':' + str(len(myList)) + '-' + 'EB Cen')
            edgeBetweenessCentralities = nx.edge_betweenness_centrality(elt)
            for k, v in edgeBetweenessCentralities.items():
                edgeBetweenessCentralityList.append(v)
            print(str(idx) + ':' + str(len(myList)) + '-' + 'CF Cen')
            cfcCentralities = nx.current_flow_closeness_centrality(elt)
            for k, v in cfcCentralities.items():
                cfcCentralityList.append(v)
            print(str(idx) + ':' + str(len(myList)) + '-' + 'SG Cen')
            subGraphCentralities = nx.subgraph_centrality(elt)
            for k, v in subGraphCentralities.items():
                sgCentralityList.append(v)
        else:
            pass
        idx += 1

populateLists(mesoList)
populateLists(thermoList)

print('Degree: ' + str(min(degList)) + '-' + str(max(degList)))
print('Clust Coeff: ' + str(min(clustCoeffList)) + '-' + str(max(clustCoeffList)))
print('Closeness Cen: ' + str(min(closenessCentralityList)) + '-' + str(max(closenessCentralityList)))
print('Degree Cen: ' + str(min(degreeCentralityList)) + '-' + str(max(degreeCentralityList)))
print('EBC: ' + str(min(edgeBetweenessCentralityList)) + '-' + str(max(edgeBetweenessCentralityList)))
print('CFCC: ' + str(min(cfcCentralityList)) + '-' + str(max(cfcCentralityList)))
print('SGC: ' + str(min(sgCentralityList)) + '-' + str(max(sgCentralityList)))
print('Done')
