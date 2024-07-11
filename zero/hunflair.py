#Add rading CSV file 
import ast
import csv
import os
import openai
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import pandas as pd
import sys

from flair.data import Sentence
from flair.nn import Classifier

output_path=sys.argv[1]
print('output_path',output_path)

#from umlsparser import UMLSParser
#umls = UMLSParser('/data1/monajati/med/UMLS/umls-extract/')

csv_path = 'gene.csv'

def fun(inp):
    dic={}
    
    sentence = Sentence(inp)
    # load the NER tagger
    tagger = Classifier.load("hunflair")

    # run NER over sentence
    tagger.predict(sentence)

    # print the sentence with all annotations
    print(sentence)
    for i in range(len(sentence.labels)):
        dic[sentence.labels[i].data_point[0].text]=sentence.labels[i].value
    
    return str(dic)
    

data = pd.read_csv(csv_path)
data['Output'] = data['text'].apply(fun)
#data.to_csv('ai_output/noRAG35_7.csv', index=False)
data.to_csv(output_path, index=False)

    
    