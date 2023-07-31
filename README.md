# NLP SuggestionMining
 An Assignment featuring tasks for feature engineering, training and evaluating a classifier for suggestion detection. You will work with the data from SemEval-2019 Task 9 subtask A to classify whether a piece of text contains a suggestion or not. 

Code Files: 
1. 22222806_Rana,ipynb

# University of Galway
## Assignment 2 - CT5120
---

This assignment involves tasks for feature engineering, training and evaluating a classifier for suggestion detection. You will work with the data from SemEval-2019 Task 9 subtask A to classify whether a piece of text contains a suggestion or not. 


Download train.csv, test_seen.csv and test_unseen.csv from the [Github](https://github.com/sharduls007/Assignment_2_CT5120) or uncomment the code cell below to get the data as a comma-separated values (CSV) file. The CSV file contains a header row followed by 5,440 rows in train.csv and 1,360 rows in test_seen.csv spread across 3 columns of data. Each row of data contains a unique id, a piece of text and a label assigned by an annotator. A label of $1$ indicates that the given text contains a suggestion while a label of $0$ indicates that the text does not contain a suggestion.

You can find more details about the dataset in Sections 1, 2, 3 and 4 of [SemEval-2019 Task 9: Suggestion Mining from Online Reviews and Forums
](https://aclanthology.org/S19-2151/).

We will be using test_seen.csv for benchmarking our model, hence it has label. On the other hand, test_unseen is used for [Kaggle](https://www.kaggle.com/competitions/nlp2022ct5120suggestionmining/overview) competition.

---

## 1. Data Preprocessing (3 methods)

* **Lowercase:** Lowercase is better to use in some cases where some algorithms are designed to take lowercase letters/words. We use lower() on each sentence to convert all the words into lowercase.

* **Punctuation Removal:** Punctuation Removal is the process of removing punctuation marks (like "!",",","?"). The idea behind this is that we do not require punctuations to carry out some of the tasks in NLP, but it can be quite nice if you use them for Sentiment Analysis. In our case, suggestions wouldn't usually contain any punctuation and will have little impact if we remove them from the dataset. We use string.punctuation library which is inbuilt in Python to check for any type of punctuation marks and if we find it, we just discard it.

* **Tokeniation:** Tokenization is the process of splitting either a query, paragraph or a document into smallest unit i.e., a word. For e.g., the sentence "I am a human" can be tokenized as "I", "am", "a", "human" and this is a good practice of NLP structures as discrete elements can be processed by the NLP model and token occurances can be used as vector representing the document. The NLTK.tokenize package has a 'word_tokenize' method which automatically converts your text into a list of tokens.

* **Stopword Removal:** Stopwords are the words which have the highest frequency in a document e.g., I, You, The, An, etc. So they provide almost no information or the meaning in any sentence. These sentences are better off removed from the sentence and hence this is called Stopword Removal. In NLTK library, we have stopwords function where there is a dictionary of stopwords defined, which we can use to filter out stopwords in a given document.

* **Lemmatization:** Lemmatization is a preprocessing method where you bring a word to its base form, e.g., running to run, better to good. Lemmatization is still in its very early stages as not every word will be converted as there are many grammatical constraints, but its a good alternative to Stemming where you just cut the words and sometimes get non-sensical words. To perform lemmatization, import WordNetLemmatizer() class from nltk,stem package. To make sure Lemmatization works well, we have to define Parts-of=speech tagging to lemmatize words better, so we used wordnet from nltk.corpus package, and tag the words into either Adjective (ADJ), Nouns (NOUN), Verbs (VERB), or Adverbs (ADV) and then send both the word and the POS of the word to lemmatize them.

## 2. Feature Engineering (I) - TF-IDF as features

In the lectures we have seen that raw counts of words and `tf-idf` scores can be useful features for a classification task. Complete the following code cell to create a suggestion detector which uses `tf-idf` scores as features for a Naïve Bayes classifier.

After applying your preprocessing steps, use the training data to train the classifier and make predictions on the test set. You **must not** use the test set for training.

If everything is implemented correctly, then you should see a single floating point value between 0 and 1 at the end which denotes the accuracy of the classifier.

## 3. Evaluation Metrics


Accuracy is not the best measure when the dataset is imbalanced, so in our case, too, it's not the best evaluation metric. Instead, we turn towards another metric to find out the reality of our model.

For any classification problems, a *confusion matrix* for each class to evaluate on the terms of **precision**, **recall** and **f1-score** is beneficial for finding out how the model works and its effectiveness.

* **Confusion Matrix:** It is a special type of error table which allows visualization of the performance of an algorithm, and is typically used in the Supervised Learning methods. It is a 2x2 matrix which shows numbers based on the values as the representation below in the table. Some of the concepts followed in the matrix are:
    1. True Positive(TP): The values which are predicted as positive and match when evaluated it with the original labels, are called True Positives.
    2. False Negative(FN): The values which are predicted as negative but the original labels are positive, are called False Negatives.
    3. False Positives(FP): The values which are predicted as positive, but the original labels state them negative, are called False Positives.
    4. True Negative(TN): The values which we predicted as negative and they matched the original labels, are called as True Negatives.

|    Total Population (P+N)   |     Positive(PP)     |    Negative(PN)      |
| :-------------------------- | -------------------: | -------------------: |
|       Positive (P)          |  True positive (TP)  |  False negative (FN) |
|       Negative (N)          |  False positive (FP) |  True negative (TN)  |

*Source: [Wikipedia -> Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)*


* Precision: Precision is a count of how many are actually positive values with respect to the values which were predicted as positive. The equation can be drawn as:
$$ Precision = \frac {TP}{TP + FP} $$ 

* Recall: Recall is a count of how many were predicted correctly with respect to all the positive classes. The equation can be drawn as:
$$ Recall = \frac {TP}{TP + FN} $$ 

* F1-Score: It is difficult to compare models with low precision and high recall or vice versa. To make them comparable, we use the F-measure or the F1-score. It is the harmonic mean which uses both precision and recall to find out the comparable way to include both precision and recall. The equation can be drawn as:
$$ F1-score = \frac {2 * Precision * Recall }{Precision + Recall} $$ 


## 4, Feature Engineering - Other Features


There are two things which I feel can change the accuracy and evaluation of the score.

1. **Bag of n-grams:** A bag of n-grams model provides not only with the words in the sentence, but also the n-grams of the sentences. There can be a different meaning to 2 or 3 words when we use together e.g., 'the dog barks' and 'the dog barked' has a difference of grammatical tense and changes the way humans interpret the sentence, so in similar way we can vectorize n-grams to get good TF-IDF values to process our model well. In our case we have used *trigrams* to fit our model. This doesn't require any additional preprocessing steps than what we have used.

2. **Multinomial Naive Bayes:** Multinomial Naive Bayes is a probabilistic learning algorithm used to classify labels by calculating probabilities and then comparing them to a threshold which is usually set by checking our for the line which separates the given classes in the mathematical space. Multinomial Naïve Bayes consider a feature vector where a given term represents the number of times it appears or very often i.e. frequency. We use the MultinomialNB() class from the *sklean.naive_bayes* package to use them. There is a hyperparameter called *alpha* which is used for smoothening of the algorithm so that it is not as steep when classifying. We are using 0.03 as the alpha value (found through just trying different values, getting better score.)


## 5. Kaggle Competition

> Head over to https://www.kaggle.com/t/1f90b74da0b7484da9647638e22d1068  
Use above classifier to predict the label for test_unseen.csv from competition page and upload the results to the leaderboard. The current baseline score is 0.36823. Make an improvement above the baseline. Please note that the evaluation metric for the competition is the f-score.

We have used the MultinomialNB instead of GaussianNB to fit our test cases as we did a trigram fit of our training case to train the model. We achieved a mean average f-score on 0.79117 on Kaggle. I feel that this method was crucial as we wanted to find whether a statement was a suggestion or not. In order to do so, we have to weight not only the words, but the words around it and I felt that having trigrams would be much better to weight and put in the model. It definitely improved when we run the training model and evaluated them. So that was the experimental motivation which led me to believe that this model may work well to surpass the baseline score. And it worked well as the models are better given the n-gram approach when finding something like a suggestion mining model where the position of the texts and their trigram weight matters more in the general context.