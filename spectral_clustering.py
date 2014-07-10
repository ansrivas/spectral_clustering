# -*- coding: utf-8 -*-
"""
Created on Sun May 18 02:20:16 2014
@author: ankur
"""



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


pos = None

def readFile():
    """reads the file and returns the indices as list of lists"""
    filename = 's.txt'
    with open(filename) as file:
        array2d = [[int(digit) for digit in line.split()] for line in file]
    return array2d
    
    
def read_file_draw_graph():
    """Creates the graph and returns the networkx version of it 'G' \n
       Takes as an input: NOTHING :P
    
    """
    global pos
    
    array2d = readFile()
    
    ROW, COLUMN = len(array2d),len(array2d[0])
    count = 0 

    G = nx.Graph()    
    
    for j in range(COLUMN):
        for i in range(ROW):
            if array2d[ROW-1-i][j] == 0:
                G.add_node(count,pos=(j,i))
                count +=1 
     
  
    pos=nx.get_node_attributes(G,'pos')
    
    
    for index in pos.keys():
        for index2 in pos.keys():
            if pos[index][0] == pos[index2][0] and pos[index][1] == pos[index2][1] -1 :
                G.add_edge(index,index2,weight=1)
            if pos[index][1] == pos[index2][1] and pos[index][0] == pos[index2][0] -1 :
                G.add_edge(index,index2,weight=1)
 
    return G
 
 
def main(numOfClusters):
        
    G = read_file_draw_graph()
    
    '''
    Create a laplacian matrix and calculate its eigenvalues and eigenvectors
    Sort the values in ascending order
    '''
    laplacianMat = nx.laplacian_matrix(G)
    eigenValues, eigenVectors = np.linalg.eig(laplacianMat)
    idx = eigenValues.argsort()   
    
    '''Sort the eigenvectors in the same ascending order based on eigenValues
    '''
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    
    listOfVectors = []
    for i in range(numOfClusters):
        listOfVectors.append(i)
    
    '''Take initial few dimensions (listOfVectors) from the eigenVectors for clustering'''    
    data = eigenVectors[:,listOfVectors]

    #Issues with kmeans2, results in empty cluster sometimes
    #warnings.warn("One of the clusters is empty. ")
    #res, labels = kmeans2(X,CLUSTERS)

    estimator= KMeans(n_clusters=numOfClusters)
    estimator.fit(data)
    labels = estimator.labels_
    
    '''the logic to draw a graph:
       The labels are actually assigned to each of the rows of our 'data' matrix,
       where rows = the corresponding vertex in graph (I started with 0,1,... )
       Get all rows with that label which map to the correspoding index in the graph 
       and simply plot it with the logic given below
    '''
    colors = 'bgrcmykw'
    for i in range(0,max(labels)+1):
        list_nodes = np.where(labels == i)[0].tolist()       
        nx.draw(G, pos, nodelist=list_nodes,node_color=colors[i],with_labels=False)
    filename =  str(numOfClusters) +"_clusters"  +".png"
    plt.savefig(filename)
    
    
    
if __name__ == "__main__":
   
    listOfClusters = [2,3,4,5,8]
    for i, val in enumerate(listOfClusters):
        main(val)

    