from bs4 import BeautifulSoup
from nltk import word_tokenize
import requests

# Scrape 8 pages in sports, arts and ...
# convert website to bag of words
def scrape_website(url):
    result = requests.get(url, verify = False)
    soup = BeautifulSoup(result.text, 'lxml')
    text = soup.get_text()
    return text

# convert raw text to bag of words
def text_to_words(text):
    lines = [line.rstrip() for line in text.splitlines() ]
    tokenedlines = [ word_tokenize(line) for line in lines ]
    bagofwords = []
    for words in tokenedlines:
        for word in words:
            if word.isalpha():
                bagofwords.append(word)

    print(bagofwords)
    return bagofwords

# implement java.lang.string hash function
def myHash(feature):
    myHashCode = 0
    n = len(feature)
    for i in range(n):
        myHashCode += ord(feature[i]) * 31 ** (n-1-i)

    return myHashCode


# apply hash function
def hash_vectorizer(features, N):
    x = [ 0 for x in range(N)]
    for feature in features:
        h = myHash(feature)
        x[h%N] += 1

    return x

# convert words into vectors.
# normalize by default using min-max normalizer.
def words_to_vector(words, normalize = True, ndim = 10000 ):
    wordvector = hash_vectorizer(words, ndim)

    if normalize:
        maxhash = max(wordvector)
        minhash = min(wordvector)
        return [ (x-minhash+0.0)/maxhash for x in wordvector]

    else:
        return wordvector