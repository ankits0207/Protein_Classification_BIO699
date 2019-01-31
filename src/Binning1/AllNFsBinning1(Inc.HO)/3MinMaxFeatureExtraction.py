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
cplList = []
csList = []
dc2List = []

def getCSGivenNX(myNx):
    listToBeReturned = []
    for n in myNx.__iter__():
        neighbors = myNx.neighbors(n)
        sumOfWeights = 0
        count = 0
        for neighbor in neighbors:
            sumOfWeights += myNx[n][neighbor]['weight']
            count += 1
        avgSumOfWeights = (sumOfWeights*1.0)/count
        listToBeReturned.append(avgSumOfWeights)
    return listToBeReturned

def get2HopDcGivenNX(myNx):
    listToBeReturned = []
    for n in myNx.__iter__():
        listOfNeighbors = []
        neighbors1 = myNx.neighbors(n)
        for n1 in neighbors1:
            listOfNeighbors.append(str(n1))
            neighbors2 = myNx.neighbors(n1)
            for n2 in neighbors2:
                listOfNeighbors.append(str(n2))
        listOfNeighbors = set(listOfNeighbors)
        listOfNeighbors.remove(n)
        listToBeReturned.append((len(listOfNeighbors)*1.0)/len(myNx))
    return listToBeReturned


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

            #CPL
            print(str(idx) + ':' + str(len(myList)) + '-' + 'CPL')
            characteristicPathLengths = nx.shortest_path_length(elt, weight='weight')
            for characteristicPathLength in characteristicPathLengths:
                extractedDict = characteristicPathLength[1]
                for k, v in extractedDict.items():
                    if v != 0:
                        cplList.append(v)
            #CS
            print(str(idx) + ':' + str(len(myList)) + '-' + 'CS')
            connectionStrengths = getCSGivenNX(elt)
            for connectionStrength in connectionStrengths:
                csList.append(connectionStrength)

            #DC2
            print(str(idx) + ':' + str(len(myList)) + '-' + 'DC2')
            DC2 = get2HopDcGivenNX(elt)
            for dc2 in DC2:
                dc2List.append(dc2)
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
print('CPL: ' + str(min(cplList)) + '-' + str(max(cplList)))
print('CS: ' + str(min(csList)) + '-' + str(max(csList)))
print('DC2: ' + str(min(dc2List)) + '-' + str(max(dc2List)))
print('Done')
