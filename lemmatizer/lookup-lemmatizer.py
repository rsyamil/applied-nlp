import sys
import re

train_file = sys.argv[1]
test_file = sys.argv[2]

#train_file = "UD_Hindi-HDTB-master/hi_hdtb-ud-train.conllu"
#test_file = "UD_Hindi-HDTB-master/hi_hdtb-ud-test.conllu"

# Counters for lemmas in the training data: word form -> lemma -> count
lemma_count = {}

# Lookup table learned from the training data: word form -> lemma
lemma_max = {}

# Variables for reporting results
training_stats = ['Wordform types' , 'Wordform tokens' , 'Unambiguous types' , 
                  'Unambiguous tokens' , 'Ambiguous types' , 'Ambiguous tokens' , 
                  'Ambiguous most common tokens' , 'Identity tokens']

training_counts = dict.fromkeys(training_stats , 0)

test_outcomes = ['Total test items' , 'Found in lookup table' , 'Lookup match' , 
                 'Lookup mismatch' , 'Not found in lookup table' , 'Identity match' , 
                 'Identity mismatch']

test_counts = dict.fromkeys(test_outcomes , 0)

accuracies = {}

### Training: read training data and populate lemma counters
train_data = open(train_file , 'r', encoding='utf8')

for line in train_data:
    
    # Tab character identifies lines containing tokens
    if re.search ('\t' , line):

        # Tokens represented as tab-separated fields
        field = line.strip().split('\t')

        # Word form in second field, lemma in third field
        form = field[1]
        lemma = field[2]
        
        if form == lemma:
            training_counts['Identity tokens'] += 1
        
        training_counts['Wordform tokens'] += 1
                
        if form in lemma_count:
            #for that particular form, check all lemmas and increment count for the right one
            arr = lemma_count[form]
            exist = False
            for idx in range(len(arr)):
                if arr[idx][0] == lemma:
                    arr[idx][1] += 1
                    exist = True
            #if that lemma doesnt exist, append
            if not exist:
                arr.append([lemma, 1])
        else:
            lemma_count[form] = [[lemma, 1]]
            training_counts['Wordform types'] += 1
            
#print({key: value for key, value in sorted(lemma_count.items(), key=lambda item: item[1])})

### Model building and training statistics
for form in lemma_count.keys():
    
    #non-ambigous ones have len of only 1
    arr = lemma_count[form]
    if len(arr) == 1:
        training_counts['Unambiguous types'] += 1
        training_counts['Unambiguous tokens'] += arr[0][1]
        lemma_max[form] = arr[0][0]
    else:
        training_counts['Ambiguous types'] += 1
        max_idx = 0
        max_val = 0
        for i in range(len(arr)):
            training_counts['Ambiguous tokens'] += arr[i][1]
            if arr[i][1] > max_val:
                max_val = arr[i][1]
                max_idx = i
        lemma_max[form] = arr[max_idx][0]
        training_counts['Ambiguous most common tokens'] += arr[max_idx][1]
        
accuracies['Expected lookup'] = (training_counts['Unambiguous tokens']+training_counts['Ambiguous most common tokens'])/training_counts['Wordform tokens']

accuracies['Expected identity'] = training_counts['Identity tokens']/training_counts['Wordform tokens']

### Testing: read test data, and compare lemmatizer output to actual lemma
test_data = open (test_file, 'r', encoding='utf8')

for line in test_data:

    # Tab character identifies lines containing tokens
    if re.search ('\t' , line):

        # Tokens represented as tab-separated fields
        field = line.strip().split('\t')

        # Word form in second field, lemma in third field
        form = field[1]
        lemma = field[2]

        test_counts['Total test items'] += 1
        
        if form in lemma_max:
            test_counts['Found in lookup table'] += 1
            if lemma_max[form] == lemma:
                test_counts['Lookup match'] += 1
            else:
                test_counts['Lookup mismatch'] += 1
        else:
            test_counts['Not found in lookup table'] += 1
            if form == lemma:
                test_counts['Identity match'] += 1
            else:
                test_counts['Identity mismatch'] += 1
                
accuracies['Lookup'] = test_counts['Lookup match']/test_counts['Found in lookup table']

accuracies['Identity'] = test_counts['Identity match']/test_counts['Not found in lookup table']

accuracies['Overall'] = (test_counts['Lookup match']+test_counts['Identity match'])/test_counts['Total test items']

### Report training statistics and test results
                
output = open ('lookup-output.txt' , 'w')

output.write ('Training statistics\n')

for stat in training_stats:
    output.write (stat + ': ' + str(training_counts[stat]) + '\n')

for model in ['Expected lookup' , 'Expected identity']:
    output.write (model + ' accuracy: ' + str(accuracies[model]) + '\n')

output.write ('Test results\n')

for outcome in test_outcomes:
    output.write (outcome + ': ' + str(test_counts[outcome]) + '\n')

for model in ['Lookup' , 'Identity' , 'Overall']:
    output.write (model + ' accuracy: ' + str(accuracies[model]) + '\n')

output.close
