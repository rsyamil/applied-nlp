#function to parse words from the input training file

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

main_input_path = sys.argv[1]
#main_input_path = "op_spam_training_data"

f_paths = [main_input_path+"/negative_polarity/deceptive_from_MTurk",
            main_input_path+"/negative_polarity/truthful_from_Web",
            main_input_path+"/positive_polarity/deceptive_from_MTurk",
            main_input_path+"/positive_polarity/truthful_from_TripAdvisor"]

labels = [["negative", "deceptive"],
         ["negative", "truthful"],
         ["positive", "deceptive"],
         ["positive", "truthful"]]

counter_table_prior = {"negative": 0,
                      "positive": 0,
                      "deceptive": 0,
                      "truthful":0 }

p_prior = {}
p_conditional = {}

bag_of_words = []
label_bag_of_words = []

LOW_F_THRESHOLD = 3                          #upper and lower limit of words to remove
HIGH_F_THRESHOLD = 1000

count = 0
for idx, p in enumerate(f_paths):
    folds = os.listdir(p)                    #read pre-defined subdir
    for f in folds:                          #may have multiple folds
        if f.startswith("fold"):
            p_f = p+'/'+f       
            reviews = (f for f in os.listdir(p_f) if f.endswith('.txt'))
            for r in reviews:                #in each fold there are reviews txt files
                count = count+1
                p_f_r = p_f+'/'+r
                p_f_r_file = open(p_f_r , 'r')
                #print(p_f_r)
                for texts in p_f_r_file:                                     #remove punctuations
                    texts = re.sub(r'[^\w\s]', ' ', texts)
                    tokens = texts.lower().split()                           #convert to lowercase
                    tokens = [w for w in tokens if not w in stopwords]       #remove stopwords
                    tokens = [w for w in tokens if not (w.isdigit()          #remove numbers
                                         or w[0] == '-' and w[1:].isdigit())]
                    bag_of_words.append(tokens)                              #save bag of words
                    label_bag_of_words.append(labels[idx])                   #save labels for bag of words
                    #for token in tokens:
                    #    print(token)
                    #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    counter_table_prior[labels[idx][0]] += 1                 #count occurences of labels
                    counter_table_prior[labels[idx][1]] += 1
                    
unique_words = list(set([item for subl in bag_of_words for item in subl]))

print("Reviews parsed : ", count)
print("No bag of words : ", len(bag_of_words))
print("No bag of words label : ", len(label_bag_of_words))
print("No words : ", len([item for subl in bag_of_words for item in subl]))
print("No unique words : ", len(unique_words))

#count unique words according to category (negative, positive) or (deceptive, truthful)
counter_table = {}
for word in unique_words:
    counter_table[word] = {"negative": 0,
                          "positive": 0,
                          "deceptive": 0,
                          "truthful":0 }

for idx, bag in enumerate(bag_of_words):
    for word in bag:
        counter_table[word][label_bag_of_words[idx][0]] += 1
        counter_table[word][label_bag_of_words[idx][1]] += 1

#calculate prior probabilities by category (negative, positive) or (deceptive, truthful)
p_prior["negative"] = counter_table_prior['negative']/(counter_table_prior['negative']+counter_table_prior['positive'])
p_prior["positive"] = counter_table_prior['positive']/(counter_table_prior['negative']+counter_table_prior['positive'])
p_prior["deceptive"] = counter_table_prior['deceptive']/(counter_table_prior['deceptive']+counter_table_prior['truthful'])
p_prior["truthful"] = counter_table_prior['truthful']/(counter_table_prior['deceptive']+counter_table_prior['truthful'])

print("Prior probability : ", p_prior)

#remove words with frequency (i.e. sum across labels, {T,F}) that are lower than LOW_F_THRESHOLD or higher than HIGH_F_THRESHOLD
words_remove = []
for w in counter_table:
    freq = counter_table[w]['negative']+counter_table[w]['positive']
    #freq = counter_table[w]['truthful']+counter_table[w]['deceptive']  equivalently
    if freq >= HIGH_F_THRESHOLD:
        words_remove.append(w)
    if freq <= LOW_F_THRESHOLD:
        words_remove.append(w)
        
for w in words_remove:
    counter_table.pop(w)
    
print("No words not between threshold : ", len(words_remove))

#add-one smoothing by category (negative, positive) or (deceptive, truthful)
for w in counter_table:
    counter_table[w]["negative"] += 0.01
    counter_table[w]["positive"] += 0.01 
    counter_table[w]["deceptive"] += 0.01 
    counter_table[w]["truthful"] += 0.01 

#sum of counts by category
w_count_negative = 0
w_count_positive = 0
w_count_deceptive = 0
w_count_truthful = 0

for w in counter_table:
    w_count_negative += counter_table[w]["negative"]
    w_count_positive += counter_table[w]["positive"]
    w_count_deceptive += counter_table[w]["deceptive"]
    w_count_truthful += counter_table[w]["truthful"]

#calculate conditional probability for each word by category 
for w in counter_table:
    p_conditional[w] = {"negative": 0,
                      "positive": 0,
                      "deceptive": 0,
                      "truthful":0 }
    p_conditional[w]["negative"] = counter_table[w]['negative']/w_count_negative
    p_conditional[w]["positive"] = counter_table[w]['positive']/w_count_positive
    p_conditional[w]["deceptive"] = counter_table[w]['deceptive']/w_count_deceptive
    p_conditional[w]["truthful"] = counter_table[w]['truthful']/w_count_truthful

#write model parameters to file
with open('nbmodel.txt', 'w', encoding='utf-8') as outf_p:
    json.dump([p_prior, p_conditional], outf_p, indent=1)


