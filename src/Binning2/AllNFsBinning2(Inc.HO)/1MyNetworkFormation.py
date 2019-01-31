import os
from Bio.PDB import *
import math
import pickle

mesoDir = 'MesoStructPDB/'
thermoDir = 'ThermoStructPDB/'
listOfMesoFiles = os.listdir(mesoDir)
listOfThermoFiles = os.listdir(thermoDir)
mesoPkl = 'MesoPkl'
thermoPkl = 'ThermoPkl'
cutoff = 6.5


class MyStructure:
    def __init__(self, pdbId, edgeList):
        self.pdbId = pdbId
        self.edgeList = edgeList


class Edge:
    def __init__(self, res1, res2, dist):
        self.res1 = res1
        self.res2 = res2
        self.dist = dist

def getEuclideanDist(currRes, nextRes):
    x1 = currRes[0]
    y1 = currRes[1]
    z1 = currRes[2]
    x2 = nextRes[0]
    y2 = nextRes[1]
    z2 = nextRes[2]
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def getListOfStructuresGivenLists(myList, dir, opFname):
    idx = 1
    myListOfStructures = []
    parser = PDBParser()
    for elt in myList:
        print(str(idx) + ':' + str(len(myList)))
        pdbStructure = parser.get_structure(elt, dir + elt)
        for chain in pdbStructure.get_chains():
            id = elt.replace('.pdb', '') + ':' + chain.get_id()
            CA_coordinates = []
            edgeList = []
            for residue in chain:
                try:
                    CA_coordinates.append(residue['CA'].get_vector())
                except:
                    pass
            for i in range(len(CA_coordinates)):
                currRes = CA_coordinates[i]
                for j in range(i+1, len(CA_coordinates)):
                    nextRes = CA_coordinates[j]
                    dist = getEuclideanDist(currRes, nextRes)
                    if dist <= cutoff:
                        edgeList.append(Edge(currRes, nextRes, cutoff-dist))
            myListOfStructures.append(MyStructure(id, edgeList))
        idx += 1
    with open(opFname, 'wb') as f:
        pickle.dump(myListOfStructures, f)

listOfMesoStructures = getListOfStructuresGivenLists(listOfMesoFiles, mesoDir, mesoPkl)
listOfThermoStructures = getListOfStructuresGivenLists(listOfThermoFiles, thermoDir, thermoPkl)
print('Done')
