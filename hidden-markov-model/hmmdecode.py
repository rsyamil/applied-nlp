import sys
import re
import os
import collections
import json

#load model parameters
with open('hmmmodel.txt', 'r', encoding='utf8') as outf_p:
    [trans, trans_end, em, unique_tags] = json.load(outf_p)
outf_p.close()
    
main_input_path = sys.argv[1]
#main_input_path = "hmm-training-data/ja_gsd_dev_raw.txt"
#main_input_path = "hmm-training-data/it_isdt_dev_raw.txt"

lines = open(main_input_path, 'r', encoding='utf8')
obss = []
tags = []

for l in lines:
    
    #get tokens separated by spaces
    tokens = l.strip().split(' ')
    obss.append(tokens)
     
print("Lines parsed : ", len(obss))
lines.close()

#assign the most likely tags for obs, from em, if doesn't exist, assign 'SYM'
'''
for obs in obss:
    tag = []
    for o in obs:
        
        #get all possible tags and their pr
        candidate_tag = []
        candidate_pr = []
        for t in em.keys():
            if o in em[t]:
                candidate_tag.append(t)
                candidate_pr.append(em[t][o])
        
        #if no possible tags
        if len(candidate_tag) == 0:
            tag.append('SYM')
        else:
            tag.append(candidate_tag[candidate_pr.index(max(candidate_pr))])
    tags.append(tag)

print("Lines of tags assigned : ", len(tags))
'''

#hidden markov model with end state

for obs in obss:
    
    #append end state
    obs = obs.copy()
    obs.append('end')
    
    #create pr matrix and backpointer
    timesteps = list(range(0, len(obs)))
    pr = dict((t1, dict((t2, 0.0) for t2 in unique_tags)) for t1 in timesteps)
    bptr = dict((t1, dict((t2, 'TERMINATED') for t2 in unique_tags)) for t1 in timesteps)
    
    #fill in both matrices
    for t, o in zip(timesteps, obs):
        
        #check if o exists in at least one of the tags
        o_exists = False
        for q in unique_tags:
            if o in em[q]:
                o_exists = True
        #print('o_exists is : ', o_exists)

        if t == 0:
            
            #if o exists in at least one of the tag
            if o_exists:
                for q in unique_tags:
                    if o in em[q]:
                        pr[0][q] = trans['start'][q]*em[q][o]
                    else: 
                        pr[0][q] = trans['start'][q]*0.0
                    bptr[0][q] = 'start'
            #if o is unseen, ignore emission pr
            else:
                for q in unique_tags:
                    pr[0][q] = trans['start'][q]
                    bptr[0][q] = 'start'
                
        elif t >= 1 and t <= (len(obs)-2):
            
            #if o exists in at least one of the tag
            if o_exists:
                for q in unique_tags:
                    candidate_tag = []
                    candidate_pr = []
                    candidate_bptr = []

                    for q_ in unique_tags:
                        if o in em[q]:
                            candidate_pr.append(pr[t-1][q_]*trans[q_][q]*em[q][o])
                            candidate_tag.append(q_) 
                        else:
                            candidate_pr.append(pr[t-1][q_]*trans[q_][q]*0.0)
                            candidate_tag.append(q_)
                        candidate_bptr.append(pr[t-1][q_]*trans[q_][q])

                    pr[t][q] = max(candidate_pr)
                    bptr[t][q] = candidate_tag[candidate_bptr.index(max(candidate_bptr))]
            #if o is unseen, ignore emission pr
            else:
                for q in unique_tags:
                    candidate_tag = []
                    candidate_pr = []

                    for q_ in unique_tags:
                        candidate_pr.append(pr[t-1][q_]*trans[q_][q])
                        candidate_tag.append(q_)

                    pr[t][q] = max(candidate_pr)
                    bptr[t][q] = candidate_tag[candidate_pr.index(max(candidate_pr))]

        else:
            for q in unique_tags:
                pr[t][q] = pr[t-1][q]*trans_end[q]
                bptr[t][q] = q
                
    #viterbi decoding
    tag = []
    max_key = max(pr[len(obs)-1], key=pr[len(obs)-1].get)
    tag.append(max_key)
    
    for t in range(len(obs)-2, 0, -1):
        max_key = bptr[t][max_key]
        tag.append(max_key)
    tag.reverse()
    tags.append(tag)

#write to output

output = open('hmmoutput.txt' , 'w', encoding='utf8')

for obs, tag in zip(obss, tags):
    out_string = ""
    for o, t in zip(obs, tag):
        out_string += o + '/' + t + ' '
    output.write(out_string.strip() + '\n')
output.close()