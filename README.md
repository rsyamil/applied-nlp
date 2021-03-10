# applied-nlp

Simple applications for natural language processing graduate course. 

### name-parser

Rule-based application to determine the full name of the first person when given a string with names of two people, such as *CHARLES DENIS AND JOHNNIE ANGELA HARRIS*. The application decides if *CHARLES DENIS*, *CHARLES DENIS HARRIS* or *CHARLES DENIS ANGELA HARRIS* is to be returned as the full name. Run as `python full-name-predictor.py dev-test.csv`. 

### lemmatizer

Application to assign tokens to lemmas using a look-up table constructed from a corpus. The application returns the most likely lemma when there is token ambiguity (i.e. a token maps to several lemmas) and if the token does not exist in the corpus, return the identity form. Run as `python lookup-lemmatizer.py UD_Hindi-HDTB-master/hi_hdtb-ud-train.conllu UD_Hindi-HDTB-master/hi_hdtb-ud-test.conllu`.

### naive-bayes

Implementation of simple Naive Bayes classification algorithm for hotel reviews. Classifies a text of hotel review (tokenized into bag of words) into either {"positive"|"negative"} and either {"truthful"|"deceptive"}. Specifically, *P("positive"|{words}) = P("positive")P({words}|"positive")*. Assuming independence (i.e. naive), the conditional probability is then *P({words}|"positive") = P("happy"|"positive")P("excellent"|"positive")....*. Add-one smoothing to account for nonexistent observation and unseen words in the test data are simply ignored. Average F1 score of 88%. To run, `python nblearn.py op_spam_training_data` followed by `python nbclassify.py op_spam_testing_data`.

### perceptron

Implementation of vanilla and average perceptron for classification of hotel reviews. Classifies a text of hotel review (tokenized into bag of words) into either {"positive"|"negative"} and either {"truthful"|"deceptive"}. Unseen words in the test data are simply ignored. F1 score of 86.7% for vanilla perceptron and 87.3% for average perceptron. To run, `python perceplearn.py op_spam_training_data` followed by `python percepclassify.py {vanillamodel.txt|averagedmodel.txt} op_spam_testing_data`.

### hidden-markov-model

Implementation of HMM for POS tagging. To learn the HMM model from a tagged corpus (of any language, in this example we will used tagged Italian and Japanese corpuses) run `python hmmlearn.py hmm-training-data/it_isdt_train_tagged.txt|ja_gsd_train_tagged.txt`. To decode a test corpus, run `python hmmdecode.py hmm-training-data/it_isdt_dev_raw.txt|ja_gsd_dev_raw.txt` and the actual answer for the test corpuses is given in `it_isdt_dev_tagged.txt|ja_gsd_dev_tagged.txt`. We use add-one smoothing for the state transition probability matrix and unseen words (i.e. do not appear for any POS tag in the emission probability matrix) are ignored (i.e. consider all possible POS tag/states). Viterbi decoding is used to trace the most likely sequence of POS tags. The accuracy for the Italian and Japanese corpuses is 93.3% and 90.99% respectively. Without HMM, baseline accuracy where the most likely tag is returned for each observation/word is 85.17% and 77.6% respectively.
