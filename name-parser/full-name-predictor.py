import sys

def calc_accuracy():
    
    key_file_lines = open('dev-key.csv', 'r')
    output_file_lines = open('full-name-output.csv', 'r')
    
    answers = []
    keys = [] 
    boths = []
    
    for l in key_file_lines:
        [both , key] = l.strip().split(',')
        keys.append(key)
        boths.append(both)
        
    for l in output_file_lines:
        [_ , ans] = l.strip().split(',')
        answers.append(ans)
        
    correct = 0.0
    wrong = 0.0
    for a, k, b in zip(answers, keys, boths):
        if (a == k):
            correct +=1
        else:
            print(b)
            print(a)
            print(k)
            print('----')
            wrong +=1
    
    print(len(answers))
    print(len(keys))
    print(correct)
    print(wrong)
    print(wrong+correct)
    print('Accuracy ' + str((correct/len(keys)*1.0)*100.0) + ' %')
    
    key_file_lines.close()
    output_file_lines.close()

def load_dictionaries():
    
    female_firsts = []
    male_firsts = []
    last_names = []
    
    female_first_lines = open('dist.female.first.txt', 'r')
    male_first_lines = open('dist.male.first.txt', 'r')
    last_name_lines = open('Names_2010Census.csv', 'r')
    
    for l in female_first_lines:
        [first_name, _, _, _] = l.strip().split()
        female_firsts.append(first_name)
    
    for l in male_first_lines:
        [first_name, _, _, _] = l.strip().split()
        male_firsts.append(first_name)
        
    for l in last_name_lines:
        items = l.strip().split(',')
        last_name = items[0]
        last_names.append(last_name)
        
    return female_firsts, male_firsts, last_names

#minus the number of words between name1 and name2 60% accuracy
def predict_with_tokens_diff(name1, name2):
    
    first_person = name1.split()
    second_person = name2.split()
    
    keep = len(second_person) - len(first_person)
    
    if keep > 0:
        to_append = second_person[-keep:]
    elif keep == 0:
        to_append = []
    else:
        to_append = []
    
    prediction = first_person + to_append
    prediction = " ".join(prediction)

    return prediction
    
#minus the number of words between name1 and name2, and use only the last word in name2 70.8% accuracy
def predict_with_tokens_diff_last(name1, name2):
    
    first_person = name1.split()
    second_person = name2.split()
    
    keep = len(second_person) - len(first_person)
    
    if keep > 0:
        to_append = second_person[-1:]
    elif keep == 0:
        to_append = []
    else:
        to_append = []
    
    prediction = first_person + to_append
    prediction = " ".join(prediction)

    return prediction
    
#final rule-based method
def predict(name1, name2, female_firsts, male_firsts, last_names):
    
    to_append = []
    
    p1 = name1.split()
    p2 = name2.split()
    
    p1 = strip_titles(p1)
    p2 = strip_titles(p2)
    
    if len(p1) >= 3:
        return name1
    
    if len(p1) == 1:
        if len(p2) == 1:
            to_append = p2 
        if len(p2) == 2:
            to_append = p2[-1:]
        if len(p2) == 3:
            to_append = strip_first_names_forp2(p2)
            if len(to_append) == 0:
                to_append = p2[-1:]
        if len(p2) >= 4:
            to_append = p2[-2:]
    
    if len(p1) == 2:
        check = strip_first_names(p1)
        if len(check) != 0:
            if has_lastname(check):
                return name1
        if len(p2) == 1:
            to_append = p2  
        if len(p2) == 2:
            to_append = p2[-1:]
        if len(p2) == 3:
            to_append = strip_first_names_forp2(p2)
            if len(to_append) == 0:
                to_append = p2[-1:]
        if len(p2) >= 4:
            to_append = p2[-2:]

    prediction = name1 + " " + " ".join(to_append)

    return prediction
    
#check if all words are first names, first word is always firstname
def strip_first_names(name):
    new_name = []
    for w in name[1:]:
        if w in female_firsts or w in male_firsts:
            pass
        else:
            new_name.append(w)
    return new_name
    
#check if all words are first names, first word is always firstname
#last word is always last name
def strip_first_names_forp2(name):
    #sometimes the last word in a long name is a surname, but also a forename
    lastidx_name = name[-1]
    new_name = []
    for w in name[1:]:
        if w in female_firsts or w in male_firsts:
            if w == lastidx_name:
                new_name.append(w)
            else:
                pass
        else:
            new_name.append(w)
    return new_name
    
#strips any titles from a full name array and return the stripped array
def strip_titles(name):
    titles = ['COLONEL', 'MAJOR', 'REVEREND', 'DOCTOR', 'PROFESSOR']
    new_name = []
    for w in name:
        if w not in titles:
            new_name.append(w)
    return new_name
    
#check if the name has a last name
def has_lastname(name):
    for w in name:
        if w in last_names:
            return True
    return False
    
if __name__ == "__main__":

    output = open('full-name-output.csv' , 'w')
    test_lines = open(sys.argv[1] , 'r')
    
    female_firsts, male_firsts, last_names = load_dictionaries()

    for line in test_lines:
        
        line = line.strip()
        [first_person , second_person] = line.split(' AND ')
        
        predicted_first_person = predict(first_person, second_person,
                                                            female_firsts, male_firsts, last_names)
        
        output.write (line + ',' + predicted_first_person + '\n')
        
    output.close

    #calc_accuracy()
