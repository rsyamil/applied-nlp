# applied-nlp

Simple applications for natural language processing graduate course. 

### name-parser

Rule-based application to determine the full name of the first person when given a string with names of two people, such as *CHARLES DENIS AND JOHNNIE ANGELA HARRIS*. The application decides if *CHARLES DENIS*, *CHARLES DENIS HARRIS* or *CHARLES DENIS ANGELA HARRIS* is to be returned as the full name. Run as `python full-name-predictor.py dev-test.csv`. 

### lemmatizer

Application to assign tokens to lemmas using a look-up table constructed from a corpus. The application returns the most likely lemma when there is token ambiguity (i.e. a token maps to several lemmas) and if the token does not exist in the corpus, return the identity form. Run as `python lookup-lemmatizer.py UD_Hindi-HDTB-master/hi_hdtb-ud-train.conllu UD_Hindi-HDTB-master/hi_hdtb-ud-test.conllu`.

### naive-bayes

Implementation of simple Naive Bayes classification algorithm for hotel reviews. Classifies a text of hotel review (tokenized into bag of words) into either {"positive"|"negative"} and either {"truthful"|"deceptive"}. Specifically, *P("positive"|{words}) = P("positive")P({words}|"positive")*. Assuming independence (i.e. naive), the conditional probability is then *P({words}|"positive") = P("happy"|"positive")P("excellent"|"positive")....*. Add-one smoothing to account for nonexistent observation and unseen words in the test data are simply ignored. Average F1 score of 88%. To run, `python nblearn.py op_spam_training_data` followed by `python nbclassify.py op_spam_testing_data`.
