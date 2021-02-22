#function to parse words from the input training file

import sys
import re
import os
import json
from collections import OrderedDict
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

bag_of_words = []
label_bag_of_words = []

#LOW_F_THRESHOLD = 10                          #upper and lower limit of words to remove
#HIGH_F_THRESHOLD = 12

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

                    
unique_words = list(set([item for subl in bag_of_words for item in subl]))   #settify (for uniqueness)

print("Reviews parsed : ", count)
print("No bag of words : ", len(bag_of_words))
print("No bag of words label : ", len(label_bag_of_words))
print("No words : ", len([item for subl in bag_of_words for item in subl]))
print("No unique words : ", len(unique_words))

#count unique words according to category (negative, positive) or (deceptive, truthful)
counter_table = OrderedDict()
for word in unique_words:
    counter_table[word] = {"negative": 0,
                          "positive": 0,
                          "deceptive": 0,
                          "truthful":0 }

for idx, bag in enumerate(bag_of_words):
    for word in bag:
        counter_table[word][label_bag_of_words[idx][0]] += 1
        counter_table[word][label_bag_of_words[idx][1]] += 1
        
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

#get the list of unique words that correspond to features
features = []
for w in counter_table:
    features.append(w)
    
print("No features : ", len(features))

#construct X_a (order as in "features") and Y_a (deceptive[-1], truthful[+1])
#construct X_b (order as in "features") and Y_b (negative [-1], positive [+1]) 
X_a, Y_a, X_b, Y_b = [], [], [], []

for idx, [b_o_w, l_b_o_w] in enumerate(zip(bag_of_words, label_bag_of_words)):
    if l_b_o_w[0] == "negative":
        Y_b.append(-1)
    else:
        Y_b.append(+1)
    if l_b_o_w[1] == "deceptive":
        Y_a.append(-1)
    else:
        Y_a.append(+1)
    feature_vector = []                 #create feature vector for each training data (exist [1], doesnt exist [0])
    for f in features:
        if f in b_o_w:                  #if the feature exists in the bag of words
            feature_vector.append(1)
        else:
            feature_vector.append(0)
    if l_b_o_w[0] == "negative" or l_b_o_w[0] == "positive":
        X_b.append(feature_vector)
    if l_b_o_w[1] == "deceptive" or l_b_o_w[1] == "truthful":
        X_a.append(feature_vector)
        
#convert everything to numpy arrays
X_a = np.array(X_a)
Y_a = np.array(Y_a)

X_b = np.array(X_b)
Y_b = np.array(Y_b)

print("Deceptive, Truthful dataset : ", X_a.shape, Y_a.shape)
print("Negative, Positive dataset : ", X_b.shape, Y_b.shape)

#RMSE for the correct sign!
def rmse(Y, Y_hat):
    return np.sqrt(np.mean((Y-np.sign(Y_hat))**2))

#learn vanilla perceptron
def train_vanilla_perceptron(X, Y, epoch):
    weights = np.zeros((X.shape[1], 1))                            #[57 x 1]
    bias = 0
    for i in range(epoch):
        np.random.seed(i)
        indexes = np.random.permutation(np.arange(0, X.shape[0]))
        for j in indexes:                                          #iterate through the entire set
            act = np.squeeze(X[j:j+1]@weights + bias)              #[1 x 1] = [1 x 57][57 x 1] + [1 x 1]
            if Y[j]*act <= 0:
                weights += np.multiply(X[j:j+1], Y[j]).T           #[57 x 1] = [57 x 1] + ([1 x 57][1 x 1]).T
                bias += Y[j]
        #calculate loss for the entire dataset
        print(rmse(Y, np.squeeze(X@weights + bias)))
    return weights, bias

#learn average perceptron
def train_average_perceptron(X, Y, epoch):
    weights = np.zeros((X.shape[1], 1))                            #[57 x 1]
    bias = 0
    weights_ave = np.zeros((X.shape[1], 1))                        #[57 x 1]
    bias_ave = 0
    w_ave = np.zeros((X.shape[1], 1))                              #[57 x 1]
    b_ave = 0
    count = 1
    for i in range(epoch):
        np.random.seed(i)
        indexes = np.random.permutation(np.arange(0, X.shape[0]))
        for j in indexes:                                          #iterate through the entire set
            act = np.squeeze(X[j:j+1]@weights + bias)              #[1 x 1] = [1 x 57][57 x 1] + [1 x 1]
            if Y[j]*act <= 0:
                weights += np.multiply(X[j:j+1], Y[j]).T           #[57 x 1] = [57 x 1] + ([1 x 57][1 x 1]).T
                bias += Y[j]
                w_ave += np.multiply(X[j:j+1], Y[j]*count).T           
                b_ave += Y[j]*count
            count += 1
        weights_ave = weights - (1.0/count)*w_ave
        bias_ave = bias - (1.0/count)*b_ave
        #calculate loss for the entire dataset
        print(rmse(Y, np.squeeze(X@weights_ave + bias_ave)))
    return weights_ave, bias_ave

print("Training vanilla perceptron for (deceptive[-1], truthful[+1])")
weights_a_vanilla, bias_a_vanilla = train_vanilla_perceptron(X_a, Y_a, epoch=15)

print("Training vanilla perceptron for (negative [-1], positive [+1])")
weights_b_vanilla, bias_b_vanilla = train_vanilla_perceptron(X_b, Y_b, epoch=15)

print("Training average perceptron for (deceptive[-1], truthful[+1])")
weights_a_average, bias_a_average = train_average_perceptron(X_a, Y_a, epoch=15)

print("Training average perceptron for (negative [-1], positive [+1])")
weights_b_average, bias_b_average = train_average_perceptron(X_b, Y_b, epoch=15)

vanilla_w_b = {"weights_a" : weights_a_vanilla.tolist(),
               "bias_a" : int(bias_a_vanilla),
               "weights_b" : weights_b_vanilla.tolist(),
               "bias_b" : int(bias_b_vanilla),
               "features" : features}
average_w_b = {"weights_a" : weights_a_average.tolist(),
               "bias_a" : int(bias_a_average),
               "weights_b" : weights_b_average.tolist(),
               "bias_b" : int(bias_b_average),
               "features" : features}

#write model parameters to file
with open('vanillamodel.txt', 'w', encoding='utf-8') as outf_p:
    json.dump([vanilla_w_b], outf_p)

with open('averagedmodel.txt', 'w', encoding='utf-8') as outf_p:
    json.dump([average_w_b], outf_p)

