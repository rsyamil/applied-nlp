#function to parse words from the input training file

import sys
import re
import os
import collections
import json

main_input_path = sys.argv[1]
#main_input_path = "hmm-training-data/ja_gsd_train_tagged.txt"
#main_input_path = "hmm-training-data/it_isdt_train_tagged.txt"

lines = open(main_input_path, 'r', encoding='utf8')
tags = []

count = 0
for l in lines:
    count += 1
    
    #get tokens separated by spaces
    tokens = l.strip().split(' ')
     
    #get unique tags by checking the entire texts
    for t in tokens:
        word, tag = t.rsplit('/', 1)
        tags.append(tag)
  
tags = list(set(tags))
    
print("Lines parsed : ", count)
print("Unique tags : ", len(tags))

#count occurences per tag, dict [tags][word]
lines.close()
lines = open(main_input_path, 'r', encoding='utf8')
em = dict((t, {}) for t in tags)

for l in lines:
    tokens = l.strip().split(' ')
    for t in tokens:
        word, tag = t.strip().rsplit('/', 1)

        #if the word is not in this tag yet
        if word in em[tag].keys():
            em[tag][word] += 1
        else:
            em[tag][word] = 1

#calculate emission probability, per tag
for tag in em:
    summ = 0.0
    for word in em[tag]:
        summ += em[tag][word]
    for word in em[tag]:
        em[tag][word] = em[tag][word]/summ

#count occurences per tag, dict [tag][tag] with 'start' and 'end' states
start_tags_end = tags.copy()
start_tags_end.insert(0, 'start')
start_tags_end.append('end')

#initialize with 0.1 for add-one smoothing
lines.close()
lines = open(main_input_path, 'r', encoding='utf8')
trans = dict((t1, dict((t2, 0.1) for t2 in start_tags_end)) for t1 in start_tags_end)

for l in lines:
    tokens = l.strip().split(' ')
    tokens.insert(0, 'start/start')
    tokens.append('end/end')
    for i in range(len(tokens)-1):
        word1, tag1 = tokens[i].strip().rsplit('/', 1)
        word2, tag2 = tokens[i+1].strip().rsplit('/', 1)
        
        #count connections to the next (outgoing) tag
        trans[tag1][tag2] += 1

#calculate transitional probability, per tag, 'end' tag will just be ignored
for tag in trans:
    summ = 0.0
    for t in trans[tag]:
        summ += trans[tag][t]
    for t in trans[tag]:
        trans[tag][t] = trans[tag][t]/summ

#delete 'end' row, i.e. nowhere to go from 'end'
_ = trans.pop('end', None)

#remove 'start' column, i.e. only outgoing arrows from 'start'
for tag in trans:
    del trans[tag]['start']

#separate all transitions into end state
trans_end = dict((t, 0) for t in tags)
for tag in trans:
    trans_end[tag] = trans[tag]['end']
    del trans[tag]['end']
    
#no empty sentence
del trans_end['start']

#write model parameters to file
lines.close()
with open('hmmmodel.txt', 'w', encoding='utf8') as outf_p:
    json.dump([trans, trans_end, em, tags], outf_p)
outf_p.close()
