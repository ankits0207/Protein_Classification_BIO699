import pickle
import networkx as nx
import pandas as pd

mesoFile = 'Meso'
mesoNxPkl = mesoFile + 'PklNx'
with open(mesoNxPkl, 'rb') as f:
    mesoList = pickle.load(f)

thermoFile = 'Thermo'
thermoNxPkl = thermoFile + 'PklNx'
with open(thermoNxPkl, 'rb') as f:
    thermoList = pickle.load(f)

pdbIdList = []
labelList = []

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

def getAvg(data):
    return (sum(data)*1.0)/len(data)

def generateFeatureVector(myList, label):
    pdbId = []
    labelList = []
    dl = []
    clcen = []
    csl = []
    dc1l = []
    dc2l = []
    sgcl = []
    ebcl = []

    idx = 1
    for elt in myList:
        if nx.is_connected(elt):
            print(str(idx) + ':' + str(len(myList)))
            degList = []
            degreeCentralityList = []
            closenessCentralityList = []
            edgeBetweenessCentralityList = []
            sgCentralityList = []

            degrees = elt.degree()
            for degree in degrees:
                degList.append(degree[1])
            fvDegree = getAvg(degList)

            closenessCentralities = nx.closeness_centrality(elt)
            for k, v in closenessCentralities.items():
                closenessCentralityList.append(v)
            fvClosenessCen = getAvg(closenessCentralityList)

            connectionStrengths = getCSGivenNX(elt)
            fvCs = getAvg(connectionStrengths)

            degreeCentralities = nx.degree_centrality(elt)
            for k, v in degreeCentralities.items():
                degreeCentralityList.append(v)
            fvDegCen = getAvg(degreeCentralityList)

            DC2 = get2HopDcGivenNX(elt)
            fvDc2 = getAvg(DC2)

            subGraphCentralities = nx.subgraph_centrality(elt)
            for k, v in subGraphCentralities.items():
                sgCentralityList.append(v)
            fvSgCen = getAvg(sgCentralityList)


            edgeBetweenessCentralities = nx.edge_betweenness_centrality(elt)
            for k, v in edgeBetweenessCentralities.items():
                edgeBetweenessCentralityList.append(v)
            fvEBCen = getAvg(edgeBetweenessCentralityList)

            pdbId.append(elt.name)
            dl.append(fvDegree)
            clcen.append(fvClosenessCen)
            csl.append(fvCs)
            dc1l.append(fvDegCen)
            dc2l.append(fvDc2)
            sgcl.append(fvSgCen)
            ebcl.append(fvEBCen)
            labelList.append(label)

        idx += 1

    myDataFrame = pd.DataFrame()
    myDataFrame['pdbId'] = pdbId
    myDataFrame['Deg'] = dl
    myDataFrame['ClosenessCen'] = clcen
    myDataFrame['ConnStr'] = csl
    myDataFrame['DC1'] = dc1l
    myDataFrame['DC2'] = dc2l
    myDataFrame['SGC'] = sgcl
    myDataFrame['EBC'] = ebcl
    myDataFrame['label'] = labelList
    return myDataFrame


mesoDf = generateFeatureVector(mesoList, '0')
mesoDf.to_csv('MesoFV.csv', index=False)
thermoDf = generateFeatureVector(thermoList, '1')
thermoDf.to_csv('ThermoFV.csv', index=False)
print('Done')
