## YOUR SCORE.PY CODE HERE ##
import pickle
import json
import re
import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import wordnet
# import my_custom_code
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from azureml.core.model import Model

def init():
    nltk.download("stopwords", quiet = True)
    nltk.download("wordnet", quiet = True)
    nltk.download("punkt", quiet = True)
    nltk.download('averaged_perceptron_tagger', quiet = True)
    
    tfidf_path = Model.get_model_path("tfidf")
    lr_path = Model.get_model_path("lr")
    
    # load the model saved in tfidf.pkl and store it in a global variable
    global lr, tfidf
    with open(lr_path, "rb") as f:
        lr = pickle.load(f)
    with open(tfidf_path, "rb") as g:
        tfidf = pickle.load(g)
    
def run(input_data):
    # convert input_data from string to JSON
    data = json.loads(input_data)['reviews']
    df = pd.DataFrame(data)
    
    df2 = preprocess(df)
    
    data = tfidf.transform(df2['join_review_body'])
    
    y_pred = lr.predict(data)
    
#     results = y_pred+1

    # return the desired response JSON
    return {
        "predictions" : y_pred.tolist()
    }

def clean_text(text):
    tags = re.compile('<.*?>') 
    text = re.sub(tags, '', text)
    #convert to lower case
    text1 = text.lower()
    #remove 's
    text2 = text1.replace("'s ", ' ')
    text2 = re.sub('\'s$', '', text2)
    #renive apostrophe character '
    text3 = text2.replace("'", "")
    return text3.strip()

def tokenize(cleaned_text):
    text1 = nltk.word_tokenize(cleaned_text)
    regex_exp = re.compile('[a-zA-Z0-9]+')
    result = sum([regex_exp.findall(s) for s in text1], [])

    return result

def lemmatize(tokens):#, stopwords = {}):
    lemmatizer = WordNetLemmatizer()
    english_stopwords = set(nltk.corpus.stopwords.words('english'))
    lemmatized_results = []

    for i in tokens:
        pos_tag = nltk.pos_tag(list([i]))[0][1]

        if pos_tag.startswith('J'):
            pos = wordnet.ADJ
        elif pos_tag.startswith('V'):
            pos = wordnet.VERB
        elif pos_tag.startswith('R'):
            pos = wordnet.ADV
        else:
            pos = wordnet.NOUN

        lemmatized_token = lemmatizer.lemmatize(word=i, pos=pos)

        if len(lemmatized_token) >= 2 and lemmatized_token not in english_stopwords:
            lemmatized_results.append(lemmatized_token)

    return lemmatized_results

def preprocess_text(text):#, stopwords = {}):
    # do not modify this function
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    return lemmatize(tokens)#, english_stopwords)

def preprocess(df):
    df = df.drop(['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'product_category','review_date'], axis=1)
    df['processed_review_body'] = df['review_body'].map(preprocess_text)
    df['verified_purchase'] = df['verified_purchase'].replace(['Y', 'N'], [1, 0])
    df['join_review_body'] = df['processed_review_body'].str.join(" ")
    return df
