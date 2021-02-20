import sys
import re
import os
import collections
import json

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

#load model parameters
with open('nbmodel.txt', 'r', encoding='utf-8') as outf_p:
    [p_prior, p_conditional] = json.load(outf_p)
    
p_posterior = {}

#parse test files, ignore folder (3-levels deep) hierarchy
main_input_path = sys.argv[1]
#main_input_path = "op_spam_testing_data"

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
                                  "negative": 1,
                                  "positive": 1,
                                  "deceptive": 1,
                                  "truthful": 1,
                                  "labela": 1,
                                  "labelb": 1}

print("Reviews (test) parsed : ", len(p_posterior))

word_count = 0
word_count_nonexistent = 0

#calculate posterior probability (p conditional)
for item in p_posterior:
    b_o_w = p_posterior[item]['bag_of_words']
    for w in b_o_w:
        word_count += 1
        if w in p_conditional:
            p_posterior[item]["negative"] *= p_conditional[w]["negative"]
            p_posterior[item]["positive"] *= p_conditional[w]["positive"]
            p_posterior[item]["deceptive"] *= p_conditional[w]["deceptive"]
            p_posterior[item]["truthful"] *= p_conditional[w]["truthful"]
        else:                                         #ignore words that dont exist
            word_count_nonexistent += 1

print("Words reviewed : ", word_count)
print("Words unseen and ignored : ", word_count_nonexistent)

#calculate posterior probability (p prior)
for item in p_posterior:
    p_posterior[item]["negative"] *= p_prior["negative"]
    p_posterior[item]["positive"] *= p_prior["positive"]
    p_posterior[item]["deceptive"] *= p_prior["deceptive"]
    p_posterior[item]["truthful"] *= p_prior["truthful"]

#argmax to assign label
for item in p_posterior:
    if p_posterior[item]["negative"] >= p_posterior[item]["positive"]:
        p_posterior[item]["labelb"] = "negative"
    else:
        p_posterior[item]["labelb"] = "positive"
    if p_posterior[item]["deceptive"] >= p_posterior[item]["truthful"]:
        p_posterior[item]["labela"] = "deceptive"
    else:
        p_posterior[item]["labela"] = "truthful"

#write to output
output = open ('nboutput.txt' , 'w')

for item in p_posterior:
    out_line = p_posterior[item]["labela"] + "\t" + p_posterior[item]["labelb"] + "\t" + p_posterior[item]["path"]
    output.write(out_line + '\n')
output.close


