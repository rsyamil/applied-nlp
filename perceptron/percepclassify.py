import sys
import re
import os
import collections
import json
import numpy as np

#from NLTK
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
             "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
             'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
             'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 
             'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
             'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
             'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
             'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
             'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
             'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
             'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
             'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 
             'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
             'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 
             'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', 
             "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
             "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 
             'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
             'won', "won't", 'wouldn', "wouldn't"]

#parse test files, ignore folder (3-levels deep) hierarchy

model_file = sys.argv[1]
main_input_path = sys.argv[2]

#model_file = "vanillamodel.txt"
#main_input_path = "op_spam_testing_data"

#load model parameters
with open(model_file, 'r', encoding='utf-8') as outf_p:
    [model_w_b] = json.load(outf_p)
     
features = model_w_b["features"]
#features = average_w_b["features"]

print("Features loaded : ", len(features))

p_posterior = {}

for root, dirs, files in os.walk(main_input_path):
    reviews = (f for f in files if f.endswith('.txt') and not (f.startswith('README')))
    for r in reviews:
        p_f_r = root+'/'+r
        p_f_r_file = open(p_f_r , 'r')
        for texts in p_f_r_file:
            texts = re.sub(r'[^\w\s]', ' ', texts)                   #remove punctuations
            tokens = texts.lower().split()                           #convert to lowercase
            tokens = [w for w in tokens if not w in stopwords]       #remove stopwords
            tokens = [w for w in tokens if not (w.isdigit()          #remove numbers
                                 or w[0] == '-' and w[1:].isdigit())]
            p_posterior[p_f_r] = {"bag_of_words": tokens,
                                  "path"        : p_f_r,
                                  "features": 1,
                                  "labela": 1,
                                  "labelb": 1}

print("Reviews (test) parsed : ", len(p_posterior))

#create feature vector for each test data point
for item in p_posterior:
    b_o_w = p_posterior[item]['bag_of_words']
    feature_vector = []                 #create feature vector for each training data (exist [1], doesnt exist [0])
    for f in features:
        if f in b_o_w:                  #if the feature exists in the bag of words
            feature_vector.append(1)
        else:
            feature_vector.append(0)
    p_posterior[item]['features'] = feature_vector

#load weights and bias
weights_a = np.asarray(model_w_b["weights_a"])
bias_a = model_w_b["bias_a"]

weights_b = np.asarray(model_w_b["weights_b"])
bias_b = model_w_b["bias_b"]

print("Weights and bias loaded for (deceptive[-1], truthful[+1]) : ", weights_a.shape, bias_a)
print("Weights and bias loaded for (negative [-1], positive [+1])  : ", weights_b.shape, bias_b)

#assign label for each test data point
for item in p_posterior:
    f = np.reshape(np.asarray(p_posterior[item]['features']), (1, len(features)))
    act_a = np.sign(np.squeeze(f@weights_a + bias_a))
    act_b = np.sign(np.squeeze(f@weights_b + bias_b))
    if act_b < 0:
        p_posterior[item]["labelb"] = "negative"
    else:
        p_posterior[item]["labelb"] = "positive"
    if act_a < 0:
        p_posterior[item]["labela"] = "deceptive"
    else:
        p_posterior[item]["labela"] = "truthful"

#write to output
output = open ('percepoutput.txt' , 'w')

for item in p_posterior:
    out_line = p_posterior[item]["labela"] + "\t" + p_posterior[item]["labelb"] + "\t" + p_posterior[item]["path"]
    output.write(out_line + '\n')
output.close




