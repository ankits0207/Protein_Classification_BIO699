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

minDeg = 1
maxDeg = 15
minClusCoeff = 0
maxClusCoeff = 1
minCloseCen = 0.03
maxCloseCen = 1
minDegCen = 0.0010
maxDegCen = 1
minEbc = 0
maxEbc = 0.66
minCfcc = 0.0001
maxCfcc = 0.75
minSgc = 1.58
maxSgc = 2339.26


def getBinnedFeatureVector(data, minVal, maxVal):
    ranges = np.linspace(minVal, maxVal, num=10)
    binDict = dict()
    binDict[0] = 0
    binDict[1] = 0
    binDict[2] = 0
    binDict[3] = 0
    binDict[4] = 0
    binDict[5] = 0
    binDict[6] = 0
    binDict[7] = 0
    binDict[8] = 0
    binDict[9] = 0
    binDict[10] = 0
    totNodes = 0
    for d in data:
        binIdx = bisect.bisect_left(ranges, d)
        count = binDict[binIdx]
        count += 1
        binDict[binIdx] = count
        totNodes += 1
    featVec = []
    for k, v in binDict.items():
        featVec.append((v*1.0)/totNodes)
    return featVec


def generateFeatureVector(myList, label):

    pdbId = []
    d = [[], [], [], [], [], [], [], [], [], []]
    clco = [[], [], [], [], [], [], [], [], [], []]
    clce = [[], [], [], [], [], [], [], [], [], []]
    dce = [[], [], [], [], [], [], [], [], [], []]
    ebc = [[], [], [], [], [], [], [], [], [], []]
    cfcc = [[], [], [], [], [], [], [], [], [], []]
    sgc = [[], [], [], [], [], [], [], [], [], []]
    labelList = []

    idx = 1
    for elt in myList:
        if nx.is_connected(elt):
            print(str(idx) + ':' + str(len(myList)))
            degList = []
            clustCoeffList = []
            closenessCentralityList = []
            degreeCentralityList = []
            edgeBetweenessCentralityList = []
            cfcCentralityList = []
            sgCentralityList = []

            degrees = elt.degree()
            for degree in degrees:
                degList.append(degree[1])
            fvDegree = getBinnedFeatureVector(degList, minDeg, maxDeg)

            clusteringCoeffs = nx.clustering(elt)
            for k, v in clusteringCoeffs.items():
                clustCoeffList.append(v)
            fvClusCoeff = getBinnedFeatureVector(clustCoeffList, minClusCoeff, maxClusCoeff)

            closenessCentralities = nx.closeness_centrality(elt)
            for k, v in closenessCentralities.items():
                closenessCentralityList.append(v)
            fvClosenessCen = getBinnedFeatureVector(closenessCentralityList, minCloseCen, maxCloseCen)

            degreeCentralities = nx.degree_centrality(elt)
            for k, v in degreeCentralities.items():
                degreeCentralityList.append(v)
            fvDegCen = getBinnedFeatureVector(degreeCentralityList, minDegCen, maxDegCen)

            edgeBetweenessCentralities = nx.edge_betweenness_centrality(elt)
            for k, v in edgeBetweenessCentralities.items():
                edgeBetweenessCentralityList.append(v)
            fvEBCen = getBinnedFeatureVector(edgeBetweenessCentralityList, minEbc, maxEbc)

            cfcCentralities = nx.current_flow_closeness_centrality(elt)
            for k, v in cfcCentralities.items():
                cfcCentralityList.append(v)
            fvCfcCen = getBinnedFeatureVector(cfcCentralityList, minCfcc, maxCfcc)

            subGraphCentralities = nx.subgraph_centrality(elt)
            for k, v in subGraphCentralities.items():
                sgCentralityList.append(v)
            fvSgCen = getBinnedFeatureVector(sgCentralityList, minSgc, maxSgc)

            pdbId.append(elt.name)
            for i in range(10):
                d[i].append(fvDegree[i])
                clco[i].append(fvClusCoeff[i])
                clce[i].append(fvClosenessCen[i])
                dce[i].append(fvDegCen[i])
                ebc[i].append(fvEBCen[i])
                cfcc[i].append(fvCfcCen[i])
                sgc[i].append(fvSgCen[i])
            labelList.append(label)

        idx += 1

    myDataFrame = pd.DataFrame()
    myDataFrame['pdbId'] = pdbId
    for i in range(10):
        myDataFrame['deg_' + str(i)] = d[i]
        myDataFrame['clco_' + str(i)] = clco[i]
        myDataFrame['clce_' + str(i)] = clce[i]
        myDataFrame['dce_' + str(i)] = dce[i]
        myDataFrame['ebc_' + str(i)] = ebc[i]
        myDataFrame['cfcc_' + str(i)] = cfcc[i]
        myDataFrame['sgc_' + str(i)] = sgc[i]
    myDataFrame['label'] = labelList
    return myDataFrame

mesoDf = generateFeatureVector(mesoList, '0')
mesoDf.to_csv('MesoFV.csv', index=False)
thermoDf = generateFeatureVector(thermoList, '1')
thermoDf.to_csv('ThermoFV.csv', index=False)
print('Done')
