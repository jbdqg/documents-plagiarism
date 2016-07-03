
# coding: utf-8

# # Equal documents detection

# In[43]:

import binascii as bh
import numpy as np
import time as tm

global DATASET_PATH
#DATASETS
    #Parte1
        #/home/datasets/unique_small.txt — has 2000 documentos; 759 KB;
        #/home/datasets/unique.txt — has 4.749.445 documentos; 1.65 GB.
    
DATASET_PATH = '/home/datasets/unique_small.txt'
global NUM_BUCKETS #next prime number from the max value of a hashed shingle (2^32)
#http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
NUM_BUCKETS= 4294967311
global DOCS_POSITION #index of each of the dataset documents
DOCS_POSITION = []


# ### Q1
# How many comparisons were made for each document?

# In[44]:

start_time = tm.time()

def hashbucket(hash_unique, a, b, c):
    return (a * hash_unique + b) % c

def hashsum(hash_unique):
    sum = 0;
    for digit in str(hash_unique):
        sum += int(digit)
    return sum

#docPosition holds the index of the beginning of each document
#to get each of the documents get tue correpondent array position
def filedocsposition():
    
    DOCS_POSITION[:] = []
    
    with open(DATASET_PATH, 'r') as f:
        DOCS_POSITION.append(f.tell())
        while f.readline():
            #print DOCS_POSITION[len(DOCS_POSITION)-1]
            DOCS_POSITION.append(f.tell())
        del DOCS_POSITION[-1]
    return DOCS_POSITION

filedocsposition()

#reads the content of the array positions for the documents of the dataset
def readDocsPosition():
    
    with open(DATASET_PATH, 'r') as f:
        for docIndex in DOCS_POSITION:
            f.seek(docIndex)
            #print f.readline()

readDocsPosition()
print `len(DOCS_POSITION)` + ' documents.'
print '\n'
print("--- %s seconds ---" % (tm.time() - start_time))


# In[45]:

#obter a hash
def hashdocs():
    
    hash_table = {}
    
    collisions = 0
    equaldocuments = 0

    np.random.seed(0)
    a = (np.random.randint(1, NUM_BUCKETS))
    b = (np.random.randint(1, NUM_BUCKETS))
        
    with open(DATASET_PATH, 'r') as f:
        if DOCS_POSITION:
            for docIndex in DOCS_POSITION:
                
                f.seek(docIndex)
                linha = f.readline()
                indice = linha.index(' ')

                hash_value = bh.crc32(linha[indice+1:linha.index('\n') if '\n' in linha else len(linha)]) & 0xffffff

                bucket = hashbucket(hash_value, a , b, NUM_BUCKETS)

                if (bucket not in hash_table):
                    hash_table[bucket] = []
                    hash_table[bucket].append(docIndex)
                                        
                else:
                    collisions += 1
                    hash_table[bucket].append(docIndex)
        
        #only considers the buckets with more than one document
        candidatePairs = sorted([cand for cand in hash_table.items() if len(cand[1]) > 1], key = len, reverse = True)
        
        hash_table.clear()
                
        return candidatePairs, collisions
start_time = tm.time()                
candidatedocs, collisions = hashdocs()

print '\n'
print("--- %s seconds ---" % (tm.time() - start_time))


# In[46]:

def findequaldocs(candidatedocs):
    
    equaldocpairs = {}
    comparisons = 0
    iterations = 0
    equaldocuments = 0
       
    with open(DATASET_PATH, 'r') as f:
        for i in range(len(candidatedocs)):

            equaldocpairs[candidatedocs[i][0]] = {}

            if len(candidatedocs[i]) > 1:
                #print len(candidatedocs[i][1])
                
                for j in range(len(candidatedocs[i][1])):

                    f.seek(candidatedocs[i][1][j])
                    linha = f.readline()
                    indice = linha.index(' ')
                    doc = linha[indice+1:linha.index('\n') if '\n' in linha else len(linha)]

                    #if (not hash_table.has_key(bucket)):
                    if (doc not in equaldocpairs[candidatedocs[i][0]]):
                        equaldocpairs[candidatedocs[i][0]][doc] = 1
                        #equaldocpairs[candidatedocs[i][0]][doc] = []
                        #equaldocpairs[candidatedocs[i][0]][doc].append(candidatedocs[i][1][j])
                    else:
                        equaldocpairs[candidatedocs[i][0]][doc] += 1
                        #equaldocpairs[candidatedocs[i][0]][doc].append(candidatedocs[i][1][j])
                    
                    #if j <> 0:
                    comparisons += 1 + len(equaldocpairs[candidatedocs[i][0]])
                    iterations += 1
                    
                #to calculate the number of comparisons
                #equaldocuments += sum([len(x) for x in equaldocpairs[candidatedocs[i][0]].values() if len(x) > 1])
                #equaldocuments += sum(equaldocpairs[candidatedocs[i][0]].values() for x in equaldocpairs[candidatedocs[i][0]].values() if len(x) > 1)
                equaldocuments += sum(x for x in equaldocpairs[candidatedocs[i][0]].values() if len(equaldocpairs[candidatedocs[i][0]].values()) == 1)
                #print sum(equaldocpairs[candidatedocs[i][0]].values())
                #equaldocuments += sum([len(x) for x in equaldocpairs[candidatedocs[i][0]].values() if len(x) > 1])
                            
            
        print 'Q1 answer:'
        print '\tWere made ' + `comparisons` + ' comparisons'
        print '\tWere made ' + `iterations` + ' iterations'
        print '\tThere are ' + `equaldocuments` + ' equal documents at the dataset'
        print '\n'
        print '\tObservation: The number of comparisons considers that when validations if a doc is already at the documents list'
        print '\t\tof a bucket it is being compared to which one of the bucket documents.'
            
    return True
start_time = tm.time()
findequaldocs(candidatedocs)
print '\n'
print("--- %s seconds ---" % (tm.time() - start_time))