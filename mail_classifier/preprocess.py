import pandas as pd
import string
import re
import wordninja
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
try:
    from nltk.stem import WordNetLemmatizer
except:
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lowercase(text):
    text=text.lower()
    return text

def long_words_break(text):
    text=' '.join(wordninja.split(text))
    return text

def lemmatize(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens]
    return lemmas

def remove_url(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www.\S+', '', text)
    return text

def remove_numbers(text):
    text = re.sub(r'\d+', ' ', text)
    return text

# def extract_urls(Email_content):
#     urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\)#,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\#(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', Email_content)
#     if urls == []:
#         return None
#     return urls

# def extract_account_num(Email_content):
#     ac_no = []
#     ac=re.findall('[^0-9][0-9]{9}[^0-9]|[^0-9][0-9]{11,17}[^0-9]|[^0-9][0-5]{1}[0-9]{9}[^0-9]', Email_content)
#     for w in ac:
#         res= w[1:len(w)-1]
#         ac_no.append(res)
#     upi = re.findall('[\w.]+@[\w]+[^\w]', Email_content)
#     return {"upi":upi, "ac_no":ac_no}

def preprocess_email(Email_content):
#     urls = extract_urls(Email_content)
#     ac_no = extract_account_num(Email_content)
    content = ' '.join((lemmatize(long_words_break(remove_numbers(lowercase(remove_url(Email_content)))))))
    return content
