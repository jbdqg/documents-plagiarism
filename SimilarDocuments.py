
# coding: utf-8

# # Detect similar documents (plagiarism)

# In[15]:

import binascii as bh
import itertools as it
import math as mt
import numpy as np
import operator as op
import sys
import time as tm

global DATASET_PATH
#DATASETS
    #Parte2
        #/home/datasets/bbc_dataset.txt — 1975 documents; 4.4 MB;
        #/home/datasets/similar.txt — 1.064.133 documents; 264 MB.
DATASET_PATH = '/home/datasets/bbc_dataset.txt'
global DOCS_POSITION #index for each of the documents
DOCS_POSITION = []
global N_BANDS #calcnumhashfunctions()
global N_ROWS #number of rows constant and equal to 5
global NUM_BUCKETS #next prime number from the max value of a hashed shingle (2^32)
#http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
NUM_BUCKETS= 4294967311
global NUM_HASHFUNCTIONS #calcnumhashfunctions()
global SHINGLE_SIZE
SHINGLE_SIZE = 3


# In[16]:

start_time = tm.time()

def hashbucket(hash_unique, a, b, c):
    return (a * hash_unique + b) % c

def hashsum(hash_unique):
    sum = 0;
    for digit in str(hash_unique):
        sum += int(digit)
    return sum

#docPosition holds the indexes where each document starts
def filedocsposition():
    
    DOCS_POSITION[:] = []
    
    with open(DATASET_PATH, 'r') as f:
        DOCS_POSITION.append(f.tell())
        while f.readline():
            DOCS_POSITION.append(f.tell())
        del DOCS_POSITION[-1]
    return DOCS_POSITION

filedocsposition()

#reads the contents of the array of the position of documents
def readDocsPosition():
    
    with open(DATASET_PATH, 'r') as f:
        for docIndex in DOCS_POSITION:
            f.seek(docIndex)
            #print f.readline()

readDocsPosition()
print `len(DOCS_POSITION)` + ' documentos.'
print '\n'
print("--- %s seconds ---" % (tm.time() - start_time))


# ## 2.1 Shingling

# In[3]:

#generates the shingles of the defined size for a document 
def shinglesgenerator(size, document):
    shingles = []
    spliteddoc = document.split()
    for i in range(len(spliteddoc)-(size-1)):
        one_shingle = ''
        for j in range(0, size):
            one_shingle += ' ' + spliteddoc[i+j]
        shingles.append(one_shingle.strip())
        
    return shingles

#converts each shingle to a 32bits hash
def shingleshasehd(shingles):
    
    for i in range(len(shingles)):
        shingles[i] = bh.crc32(shingles[i]) & 0xffffffff
        
#returns the shigles of a dataset
def get_datasetshingles(dataset, shingle_size, hash_shingles):
    
    allShingles = {}
    
    with open(dataset, 'r') as f:
        for docIndex in DOCS_POSITION:
            
            f.seek(docIndex)
            linha = f.readline()
            
            indice = linha.index(' ')
            shingles = shinglesgenerator(shingle_size, linha[indice+1:linha.index('\n') if '\n' in linha else len(linha)])
            
            if hash_shingles == True:
                shingleshasehd(shingles)
                
            #allShingles.append(shingles)
            #allShingles.append((docIndex ,shingles))
            allShingles[docIndex] = shingles
                        
    return allShingles

#orders the dataset list based on the size of the shingles
def sortshingles(dataset_shingles):
    #return dataset_shingles.sort(key=lambda x: len(x[1]))
    return sorted(dataset_shingles, key=lambda k: len(dataset_shingles[k]), reverse=True)
        
#total number of shingles on a dataset
def totaldatasetshingles(dataset_shingles):
    return sum(len(shingles) for shingles in dataset_shingles.values())

#average number of character per word
def avg_wordsize(dataSet):
        
    count_documents = 0
    sum_docwords = 0
    
    with open(dataSet, 'r') as f:
        for linha in f:
            indice = linha.index(' ')
            sum_docwords += float(sum(len(a_shingle) for a_shingle in linha[indice:].split()) / len(linha[indice:].split()))
            count_documents += 1        
            
    return int(sum_docwords / count_documents)

#average number of characters per shingle
def avg_textshinglesize(dataSet, size):
    avg_shingle = ''
    
    #each shingle has N words, so the average number of character per shingle is = a avg_wordsize * N
    for i in range(size * avg_wordsize(dataSet)):
        avg_shingle += 'x'
    
    #for each average number of characters has to be added the size of ' ' that each shingle has N-1 ' '
    for i in range(size - 1):
        avg_shingle += ' '
    
    #sizeof returns the number of bytes
    return sys.getsizeof(avg_shingle.encode('utf8'))


# ### Q3: Whats the min, average and max number of shingles per document?

# In[4]:

#min number of shingles per document of a dataset
def mindatasetshingles(sorted_dataset_shingles):
    return len(docShingles[sorted_dataset_shingles[-1]])

#max number of shingles per document of a dataset
def maxdatasetshingles(sorted_dataset_shingles):
    return len(docShingles[sorted_dataset_shingles[0]])

#average number of shingles per document of a dataset
def avgdatasetshingles(dataset_shingles):
    return sum(len(shingles) for shingles in dataset_shingles.values()) / len(dataset_shingles)

start_time = tm.time()
#obter uma lista com os shingles do dataset
docShingles = get_datasetshingles(DATASET_PATH, SHINGLE_SIZE, True)

docShinghlesKeys = sortshingles(docShingles)

#sets the number of N_BANDS, N_ROWS e NUM_HASHFUNCTIONS
avgshingles = avgdatasetshingles(docShingles)
num_shingles = (sum(len(shingles) for shingles in docShingles.values()))
if DATASET_PATH == '/home/datasets/bbc_dataset.txt':
    N_ROWS = 5
    NUM_HASHFUNCTIONS = int(mt.floor((num_shingles / len(DOCS_POSITION)) / 100.0) * 100)
    N_BANDS = NUM_HASHFUNCTIONS / N_ROWS
if DATASET_PATH == '/home/datasets/similar.txt':
    N_ROWS = 2
    N_BANDS = 5
    NUM_HASHFUNCTIONS = 10 

#N_BANDS = 30
#NUM_HASHFUNCTIONS = 150

print 'Q3:'
print '\tMin of Shingles: ' + `mindatasetshingles(docShinghlesKeys)`
print '\tMax of Shingles: ' + `maxdatasetshingles(docShinghlesKeys)`
print '\tAverage of Shingles: ' + `avgshingles`

print '\n'
print("--- %s seconds ---" % (tm.time() - start_time))


# ### Q4
# If all shingles are kept in memory, whats the amount of saved space by using a hash instead fo the text shingles?
# It is used the average length of words as estimative.

# In[5]:

#Q4

start_time = tm.time()

shingles = docShingles

#total number of shingles
total_shingles = totaldatasetshingles(shingles)
    
#used memory for a hashed shingle
bytes_hashedshingle = sys.getsizeof(shingles[0][0])

#average memory for a text shingle
bytes_avgtextshingle = avg_textshinglesize(DATASET_PATH, SHINGLE_SIZE)

total_hashedshingle = total_shingles * bytes_hashedshingle
total_avgtextshingle = total_shingles * bytes_avgtextshingle

bytes_saved =  total_avgtextshingle - total_hashedshingle
mb_saved = float(bytes_saved/1024/1024)
percentage_saved = (100 - ((total_hashedshingle * 100)/total_avgtextshingle))

#memory saved
print 'Q4:'
print '\tThere was a saving of ' + `percentage_saved ` + '% of space '
print '\toccupied (' + `bytes_saved` + 'bytes or aprox ' + `mb_saved` + 'Mb) '
print '\tby using hashed shingles insted of text shingles'

print '\n'
print("--- %s seconds ---" % (tm.time() - start_time))


# ## 2.2 Minhashing

# In[6]:

#calculates a and b random numbers between 2^32 scale
def calcAandB():

    aVals = []
    bVals = []

    #to guarantee that the experience can be repeated using the same values
    np.random.seed(NUM_HASHFUNCTIONS)

    for i in range(NUM_HASHFUNCTIONS):
        aVals.append(np.random.randint(1, NUM_BUCKETS))
        bVals.append(np.random.randint(1, NUM_BUCKETS))
    
    return aVals, bVals

#to calculate jaccard similarity
def jaccardsim(signatureDoc1, signatureDoc2):
    return float(len(set(signatureDoc1).intersection(set(signatureDoc2)))) / float(len(set(signatureDoc1).union(set(signatureDoc2))))


# ### Q5
# Comment about the considered number of hash functions.
# 
# #### Resposta Q5:
# 
# The number of hash functions (n) to consider may be chosen randomly, in which the book of Mining of Massive Datasets may be a number between 100 and a few hundred.
#
# This number corresponds to the number of permutations that will simulate and based on the characteristic matrix is ​​the set of documents in columns, n corresponds to a small part of the number of rows in the matrix representing the elements (shingles) documents .
#
# The number of hash functions defines the size of the signature, wherein the higher will be the signature of each document, the lower the possibility of collisions of different documents, but the more time and space occupied by the processing.
#
# The change in the number of bands (B) and rows (R) to be considered for the LSH will depend on the number of hash functions that consider the Minhashing wherein b = n * r.
#
# I chose to calculate the number of hash functions based on the number of shingles and the number of dataset document (nr nr = hash functions shingles / nr documents)

# In[7]:

#MINHASHING
def minhashsignatures():

    docs_local_position = DOCS_POSITION
    
    docs_local_shingles = docShingles
        
    randA, randB = calcAandB()
    
    def signpos(x): return min((randA[x] * oneShingle + randB[x]) % NUM_BUCKETS for oneShingle in ldocShingles)
    
    allSignatures = {}
    
    #to get each document signature
    with open(DATASET_PATH, 'r') as f:
        #for linha in f:
        for docIndex in docs_local_position:
            
            f.seek(docIndex)
            linha = f.readline()
            
            indice = linha.index(' ')
            
            #generate signature
            ldocShingles = docs_local_shingles[docIndex]
            
            #with list comprehensions
            #[allSignatures[docIndex].append((min((randA[i] * oneShingle + randB[i]) % NUM_BUCKETS for oneShingle in ldocShingles))) for i in range(NUM_HASHFUNCTIONS)]
            
            #with for
            #for i in range(0, NUM_HASHFUNCTIONS):
            #   minhashValue = False
            #   for oneShingle in docShingles:

            #        shingleValue = (randA[i] * oneShingle + randB[i]) % NUM_BUCKETS

            #        if minhashValue != False:
            #            if shingleValue < minhashValue:                            
            #                minhashValue = shingleValue
            #        else:
            #            minhashValue = shingleValue
                        
            #    allSignatures[docIndex].append(minhashValue)

            #with map
            allSignatures[docIndex] = map(signpos, range(NUM_HASHFUNCTIONS))
                 
    return allSignatures

start_time = tm.time()

minhasedSignatures = minhashsignatures()

print '\n'
print("--- %s seconds ---" % (tm.time() - start_time))


# ## 2.3 Locality-Sensitive hashing

# ### Q6
# How many buckets contain more than one document? What is the average number of documents per bucket?

# In[8]:

start_time = tm.time()

nbands = N_BANDS
nrows = N_ROWS

def splitvector(vector, nrows):
    return [vector[i:i+nrows] for i in range(0, len(vector), nrows)]

randA, randB = calcAandB()

splitedRandA = splitvector(randA, nrows)

allBuckets = [{} for x in range(nbands)]

def signaturebuckets(docSignature, nbands, nrows, splitedRandA, randB, docPosition):
    
    signatureBuckets = []
        
    splitedSignature = splitvector(docSignature, nrows)
    
    for band in range(nbands):
        bValue = randB[band]
        axValue = 0
       
        for i in range(len(splitedSignature[band])):
            axValue += (splitedSignature[band][i] * splitedRandA[band][i])
           
        bucket = ( (axValue + randB[band]) % NUM_BUCKETS)
    
        if bucket in allBuckets[band]:
            allBuckets[band][bucket].append(docPosition)
            # the row band was mapped to a bucket that had another element,
            # so it is not necessary to continue to calculate the buckets,
            # it will be necessary to test the similarity
            #break
        else:
            allBuckets[band][bucket] = []
            allBuckets[band][bucket].append(docPosition)
    
    return signatureBuckets

localminhashedsignatures = minhasedSignatures

for key in localminhashedsignatures:
    signaturebuckets(localminhashedsignatures[key], nbands, nrows, splitedRandA, randB, key)

sortedBuckets = []

sumMoreOneDoc = [0] * len(allBuckets)
avgBucketDoc = 0
sumAllBuckets = 0 # contains the sum of bucket bands
sumCandidateDocBucket = 0 #contains the sum of buckets documents with more than one document

bucketsAvg = []

#sort by number of documents to be first those who want to compare

for i in range(len(allBuckets)):
    sortedBuckets.append(sorted(allBuckets[i].items(), key= lambda s: len(s[1]), reverse = True))
    #summing the buckets with more than one document
    #print allBuckets[i].items()
    #break
    sumMoreOneDoc[i] += sum(1 for j in allBuckets[i].items() if len(j[1]) > 1)
    sumCandidateDocBucket += sum(len(j[1]) for j in allBuckets[i].items() if len(j[1]) > 1)
    sumAllBuckets += len(allBuckets[i].items())
    bucketsAvg.append((len(allBuckets[i].items()), (float(len(DOCS_POSITION)) / len(allBuckets[i].items()))))

print 'Q6:'
print '\tThere are ' + `sum(sumMoreOneDoc)` + ' buckets with more than one document'
print '\tObs: the counting is done considering each bucket of each of the bands ( were considered ' + `N_BANDS` + ' bands with ' + `N_ROWS` + ' rows each)'
print '\tNumber of buckets with more than one document per band: ' + `sumMoreOneDoc`
print '\tEach bucket has an average of ' + `sum((float(bucketsAvg[j][0])/sumAllBuckets) * bucketsAvg[j][1] for j in range(len(bucketsAvg)))` + ' documents'
print '\tObs: weighted average is shown as the bands may not all have the same number of buckets'
print '\tEach bucket with more than one document has an average of ' + `float(sumCandidateDocBucket) / sum(sumMoreOneDoc)` + ' documents'

print '\n'
print("--- %s seconds ---" % (tm.time() - start_time))


# ### Q7
# How many comparisons were made ? This number is compared to the number of direct comparisons between documents.

# In[9]:

#function that receives a bucket and returns sets of documents for comparison
def calcbucketpairs(bucket):
    
    bucketPairs = []
    
    for i in range(len(bucket[1])):
        for j in range(i+1, len(bucket[1])):
            bucketPairs.append((bucket[1][i], bucket[1][j]))
            
    return bucketPairs


# In[10]:

nbands = N_BANDS
nrows = N_ROWS

start_time = tm.time()

documentsToCompare = set()

#place at the document to compare only the buckets more than one document
for i in range(len(sortedBuckets)):
    for j in range(len(sortedBuckets[i])):
        if len(sortedBuckets[i][j][1]) > 1:
            documentsToCompare.add(tuple(sortedBuckets[i][j][1]))          
            
def lshSimilarities(docIndexesToCompare, thresholdVal):
    
    similarDocs = set()
    ncomparisons = 0
    compareddocs = []
        
    with open(DATASET_PATH, 'r') as f:
        for docindexes in docIndexesToCompare:
            
            pairs = list(it.combinations(docindexes, 2))
            #ncomparisons += len(pairs)
            
            #print pairs
            
            equalPair = []
            
            for j in range(len(pairs)):
                
                #if (pairs[j] not in compareddocs):
                
                ncomparisons += 1

                similarity = jaccardsim(minhasedSignatures[pairs[j][0]], minhasedSignatures[pairs[j][1]])

                if (similarity >= thresholdVal):
                    similarDocs.add(
                        (pairs[j][0], 
                         pairs[j][1], 
                         similarity)
                    )
                #compareddocs.append(pairs[j])
    
    return list(similarDocs), ncomparisons

threshold = float(1)/nbands**(float(1)/nrows)


docsSimilarities, ncomparisons = lshSimilarities(documentsToCompare, threshold)

print 'Q7:'
print '\tWere made ' + `ncomparisons` + ' documents comparison'
print '\tIf were made direct comparisons would be required to be made C(' + `len(DOCS_POSITION)` + ',2)'
print '\tThe thershold considered was ' + `threshold` + ' (1/bands^(1/rows))'
print '\tThe probability of a pair is a pair candidate is  ' + `1-(1-threshold**N_ROWS)**N_BANDS` + ' (1-(1-threshold^rows)^bands)'
print '\n'
print("--- %s seconds ---" % (tm.time() - start_time))


# In[11]:

#prints the content of a document
def printdoc(doc1Index, doc2Index):
    
    pos1 = doc1Index
    pos2 = doc2Index
    
    with open(DATASET_PATH, 'r') as f:
        f.seek(pos1)
        linha = f.readline()

        indice = linha.index(' ')
        print '----------------------------------------'
        print (linha[indice+1:linha.index('\n') if '\n' in linha else len(linha)])
        
        f.seek(pos2)
        linha = f.readline()

        indice = linha.index(' ')
        print '----------------------------------------'
        print (linha[indice+1:linha.index('\n') if '\n' in linha else len(linha)])


# In[12]:

#função top_N

#returns the n pairs of most similar documents dataset
def top_n(npairs, datasetsimilarities):
    
    #n more similar documents where each n is a pair
    datasetsimilarities.sort(key = lambda s: s[2], reverse = True)
    
    bucketPairs = datasetsimilarities
    
    topSimilar = []
    
    if len(bucketPairs) > 0:
        #tests the similarity of each pair and the most similar npairs are returned
        with open(DATASET_PATH, 'r') as f:
            for i in range(0,npairs):
                f.seek(bucketPairs[i][0])
                linha = f.readline()
                docNum1 = linha[:linha.index(' ')]
                f.seek(bucketPairs[i][1])
                linha = f.readline()
                docNum2 = linha[:linha.index(' ')]
                
                sim_jaccard = jaccardsim(minhasedSignatures[bucketPairs[i][0]], minhasedSignatures[bucketPairs[i][1]])
                
                topSimilar.append(
                    (docNum1, 
                     docNum2, 
                     sim_jaccard)
                )
                
                print 'The documents ' + `docNum1` + ' and ' + `docNum2` + ' have a similarity of ' + `sim_jaccard`
                
    return topSimilar


print top_n(2, list(docsSimilarities))


# In[13]:

def getline(numDoc):
    
    docText = ''
    
    with open(DATASET_PATH, 'r') as f:
        for linha in f:
            indice = linha.index(' ')
            if linha[:indice] == `numDoc`:
                docText = linha[linha.index(' ')+1:linha.index('\n') if '\n' in linha else len(linha)]
                break
    
    if (docText == ''):
        docText = 'The doc number ' + `numDoc` + ' does not exist on the dataset'
    
    return docText

print getline(8)


# ### Q9
# Choose a pair of similar documents for a Jaccard similarity understood in the following ranges:
# - \> 0.9
# - 0.6 a 0.8
# - < 0.5

# In[14]:

#Q9

start_time = tm.time()

docsSimilarities = lshSimilarities(documentsToCompare, 0)

print 'Q9:'

def similarDocsInterval (similarDocs, interval, intervalLabel):
    with open(DATASET_PATH, 'r') as f:
        
        outputValue = ''
        
        try:
            sim1 = next(item for item in docsSimilarities[0] if eval(interval))
            
            f.seek(sim1[0])
            linha = f.readline()
            docNum1 = linha[:linha.index(' ')]
            doc1 = linha[linha.index(' ')+1:linha.index('\n') if '\n' in linha else len(linha)]

            f.seek(sim1[1])
            linha = f.readline()
            docNum2 = linha[:linha.index(' ')]
            doc2 = linha[linha.index(' ')+1:linha.index('\n') if '\n' in linha else len(linha)]
                        
            outputValue = '\n'
            outputValue += 'Similarity ' + intervalLabel
            outputValue += '\n'
            outputValue += '\t' + `docNum1` + ' - ' + doc1
            outputValue += '\n'
            outputValue += '\t' + `docNum2` + ' - ' + doc2
            outputValue += '\n'
            outputValue += '\tDocuments ' + `docNum1` + ' and ' + `docNum2` + ' have an estimated similarity of ' + `sim1[2]`
            outputValue += '\n'
            outputValue += '\tDocuments ' + `docNum1` + ' and ' + `docNum2` + ' have a true similarity of ' + `jaccardsim(shinglesgenerator(3, doc1), shinglesgenerator(3, doc2))`
            
        except StopIteration:
            outputValue += '\n'
            outputValue += 'No pair of documents found with the desired similarity'
            outputValue += '\n'
            pass
        
    return outputValue

print similarDocsInterval(docsSimilarities, '(item[2] > 0.9)', '> 0.9')    
print similarDocsInterval(docsSimilarities, '(0.6 <= item[2] <= 0.8)', '>= 06 e <= 0.8')
print similarDocsInterval(docsSimilarities, '(item[2] < 0.5)', '< 0.5')

print '\n'
print 'There are differences between the estimated and the true similarity similarity of Jaccard, '
print 'because the estimated considers only a fraction of the lines (subscription shares ) totals that make up each document. '

print '\n'
print("--- %s seconds ---" % (tm.time() - start_time))