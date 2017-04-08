import nltk
import string
import pandas as pd
import numpy as np

from collections import Counter
from sklearn import preprocessing as pf

data = pd.read_csv('../lynch.csv')

getTop=100
getTopN=20

def get_tokens(file):
   with open(file, 'r') as shakes:
    text = shakes.read()
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    no_punctuation = lowers.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

def get_tokens2(file):
   with open(file, 'r') as shakes:
    text = shakes.read()
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    no_punctuation = lowers.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    tokens2=[]
    for i in range(len(tokens)-1):
    	tokens2.append(tokens[i]+' '+tokens[i+1])    	
    return tokens2

def get_tokens3(file):
   with open(file, 'r') as shakes:
    text = shakes.read()
    lowers = text.lower()
    #remove the punctuation using the character deletion step of translate
    no_punctuation = lowers.translate(None, string.punctuation)
    tokens = nltk.word_tokenize(no_punctuation)
    tokens3=[]
    for i in range(len(tokens)-2):
    	tokens3.append(tokens[i]+' '+tokens[i+1]+' '+tokens[i+2])    	
    return tokens3

def line_get_tokens(str):
   return str.split()

def line_get_tokens2(str):
    tokens=str.split()
    tokens2=[]
    for i in range(len(tokens)-1):
    	tokens2.append(tokens[i]+' '+tokens[i+1])    	
    return tokens2

def line_get_tokens3(str):
	tokens=str.split()
	tokens3=[]
	for i in range(len(tokens)-2):
		tokens3.append(tokens[i]+' '+tokens[i+1]+' '+tokens[i+2])    	
	return tokens3

tokens2 = get_tokens2('../lynch.txt')
tokens = get_tokens('../lynch.txt')
tokens3 = get_tokens3('../lynch.txt')
count = Counter(tokens)
hashTag1 = count.most_common(getTop)
count2 = Counter(tokens2)
hashTag2 = count2.most_common(getTop)
count3 = Counter(tokens3)
hashTag3 = count3.most_common(getTop)

'''
tokensN = get_tokens('txt2.txt')
tokensN2 = get_tokens('txt2.txt')
tokensN3 = get_tokens('txt2.txt')
countN = Counter(tokensN)
hashTagN1 = countN.most_common(getTopN)
countN2 = Counter(tokensN2)
hashTagN2 = countN2.most_common(getTopN)
countN3 = Counter(tokensN3)
hashTagN3 = countN3.most_common(getTopN)
'''

unigram=[]
two_gram=[]
three_gram=[]

for i in range(len(hashTag1)):
	unigram.append(hashTag1[i][0])

for i in range(len(hashTag2)):
	two_gram.append(hashTag2[i][0])

for i in range(len(hashTag3)):
	three_gram.append(hashTag3[i][0])
'''
unigramN=[]
two_gramN=[]
three_gramN=[]

for i in range(len(hashTagN1)):
	unigramN.append(hashTagN1[i][0])

for i in range(len(hashTagN2)):
	two_gramN.append(hashTagN2[i][0])

for i in range(len(hashTagN3)):
	three_gramN.append(hashTagN3[i][0])
'''
#load data
feature=[]
label=list(data['xinsong'])
for i in range(len(data)):
	feature.append([])
	count = Counter(line_get_tokens(data['text'].iloc[i]))
#	countN = Counter(line_get_tokens(data['username'].iloc[i]))
	try:
		count2 = Counter(line_get_tokens2(data['text'].iloc[i]))
	except:
		count2 = None
	try:
		count3 = Counter(line_get_tokens3(data['text'].iloc[i]))
	except:
		count3 = None
	for j in range(len(unigram)):
		if(count.get(unigram[j])==None):
			feature[i].append(0)
		else:
			feature[i].append(count.get(unigram[j]))
	for j in range(len(two_gram)):
		if(count2.get(two_gram[j])==None):
			feature[i].append(0)
		else:
			feature[i].append(count2.get(two_gram[j]))
	for j in range(len(three_gram)):
		if(count3.get(three_gram[j])==None):
			feature[i].append(0)
		else:
			feature[i].append(count3.get(three_gram[j]))
'''
	try:
		countN2 = Counter(line_get_tokens2(data['username'].iloc[i]))
	except:
		countN2 = None
	try:
		countN3 = Counter(line_get_tokens3(data['username'].iloc[i]))
	except:
		countN3 = None

'''


'''
	for j in range(len(unigramN)):
		if(countN.get(unigramN[j])==None):
			feature[i].append(0)
		else:
			feature[i].append(countN.get(unigramN[j]))
	for j in range(len(two_gramN)):
		if(countN2.get(two_gramN[j])==None):
			feature[i].append(0)
		else:
			feature[i].append(countN2.get(two_gramN[j]))
	for j in range(len(three_gramN)):
		if(countN3.get(three_gramN[j])==None):
			feature[i].append(0)
		else:
			feature[i].append(countN3.get(three_gramN[j]))
'''
feature = np.asarray(feature)

def load_trainingdata():
	dataF=feature[0:800]
	dataL=label[0:800]
	return dataF, dataL

def load_testingdata():
	dataF=feature[800:1000]
	dataL=label[800:1000]
	return dataF, dataL