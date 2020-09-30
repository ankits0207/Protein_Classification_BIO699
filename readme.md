Author: Ankit Sharma(BIO699) - MT16121 - ankit16121@iiitd.ac.in

Department of Computational Biology, Indraprastha Institute of Information Technology, IIIT-Delhi
*****File description*****
1. Supplementary_Data_1.xlsx - This file contains the complete list of thermophilic and mesophilic 
organisms considered for classification.
2. Supplementary_Data_2.xlsx - This file contains the list of thermophilic proteins from the finalized
dataset.
3. Supplementary_Data_3.xlsx - This file contains the list of mesophilic proteins from the finalized
dataset.
*****Runtime requirements: Python 3.6, Biopython 1.73 networkX 2.2 and other usual machine learning libraries*****

#### Some important points:

1. Source codes are present in src for replication.
2. Directory names are in correspondance to their experiments.
3. Python scripts contain their execution sequence number as the first character in their name.
4. PDB files needed in 'Binning1' and 'Binning2' are placed separately inside 'Data' directory under 'MesoStructPDB' and 'ThermoStructPDB'.
5. Network files 'MesoPklNx' and 'ThermoPklNx' are also placed inside 'Data' directory.
6. Random search and grid search sections are to be run one after the other. Comment grid search while running random search and vice-versa.

Data directory: https://drive.google.com/drive/folders/12RRmCPtlrBbEENc_n99RQIbK4aCzfgEF?usp=sharing

Note: During replication, the runtime would dump files in local directories for subsequent stages, please ensure availability of atleast 2GB disk space. Charts directory contains the code for plotting results.

Binning 1 = DSA
Binning 2 = DSF
GD - Gao ding's work, Gao, X. and Ding, Y. (2017). Using the residue interaction network improve the classification of thermophilic and mesophilic proteins. Current Bioinformatics, 12(3), 249–257.

