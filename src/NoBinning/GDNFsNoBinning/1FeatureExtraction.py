import pickle
import networkx as nx
import numpy as np
import bisect
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

def getAvg(data):
    return (sum(data)*1.0)/len(data)

def generateFeatureVector(myList, label):
    pdbId = []
    labelList = []
    dl = []
    cluscol = []
    clocenl = []
    csl = []
    cpll = []

    idx = 1
    for elt in myList:
        if nx.is_connected(elt):
            print(str(idx) + ':' + str(len(myList)))
            degList = []
            clustCoeffList = []
            closenessCentralityList = []
            cpllist = []

            degrees = elt.degree()
            for degree in degrees:
                degList.append(degree[1])
            fvDegree = getAvg(degList)

            clusteringCoeffs = nx.clustering(elt)
            for k, v in clusteringCoeffs.items():
                clustCoeffList.append(v)
            fvClusCoeff = getAvg(clustCoeffList)

            closenessCentralities = nx.closeness_centrality(elt)
            for k, v in closenessCentralities.items():
                closenessCentralityList.append(v)
            fvClosenessCen = getAvg(closenessCentralityList)

            characteristicPathLengths = nx.shortest_path_length(elt, weight='weight')
            for characteristicPathLength in characteristicPathLengths:
                extractedDict = characteristicPathLength[1]
                for k, v in extractedDict.items():
                    if v != 0:
                        cpllist.append(v)
            fvCpl = getAvg(cpllist)

            connectionStrengths = getCSGivenNX(elt)
            fvCs = getAvg(connectionStrengths)

            pdbId.append(elt.name)
            dl.append(fvDegree)
            cluscol.append(fvClusCoeff)
            clocenl.append(fvClosenessCen)
            cpll.append(fvCpl)
            csl.append(fvCs)
            labelList.append(label)

        idx += 1

    myDataFrame = pd.DataFrame()
    myDataFrame['pdbId'] = pdbId
    myDataFrame['Deg'] = dl
    myDataFrame['ClusCoeff'] = cluscol
    myDataFrame['ClosenessCen'] = clocenl
    myDataFrame['CPL'] = cpll
    myDataFrame['ConnStr'] = csl
    myDataFrame['label'] = labelList
    return myDataFrame

mesoDf = generateFeatureVector(mesoList, '0')
mesoDf.to_csv('MesoFV.csv', index=False)
thermoDf = generateFeatureVector(thermoList, '1')
thermoDf.to_csv('ThermoFV.csv', index=False)
print('Done')
