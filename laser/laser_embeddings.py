from laserembeddings import Laser
import pandas as pd
import json
import numpy as np
import csv

laser = Laser()

def get_language_stopwords():
    language_stopwords = {}

    with open('stopwords.csv', mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # Skip the header row
        next(reader)
        
        # Read each row and populate the dictionary
        for row in reader:
            lang_code = row[0]
            stopword_list = row[1].split(', ')  # Split the comma-separated stopwords back into a list
            language_stopwords[lang_code] = stopword_list
    return language_stopwords

def get_embeddings(texts, code):
    embeddings = laser.embed_sentences(texts, lang=code)
    return embeddings

# read translations.csv into a dataframe
translations = pd.read_csv('../translations.csv')
language_stopwords = get_language_stopwords()

# add a column called "sentence" to the dataframe that stores the 'en' column
translations['sentence'] = translations['en']

# UNCOMMENT FOR NO STOPWORDS, translations should only have columns with all stopword_languages and "sentence"
translations = translations[list(language_stopwords.keys()) + ['sentence']]

translations.set_index('sentence', inplace=True)

# key is sentence in language, value is embedding
embeddings = {}

# COMMENT LINE BELOW FOR WITH STOPWORDS
# translations.drop(columns=['ja'], inplace=True)

# get embeddings for each column other than 'sentence' column, and replace the sentence with the get_embeddings() value
for lang_code in translations.columns:
    sentences = translations[lang_code].to_list()
    # UNCOMMENT FOR REMOVING STOPWORDS, remove all stopwords from the sentence for that lang_code
    sentences_without_stopwords = [' '.join([word for word in sentence.split() if word.lower() not in language_stopwords[lang_code]]) for sentence in sentences]
    vectors = get_embeddings(sentences_without_stopwords, lang_code).tolist()
    for i, embedding in enumerate(vectors):
        sentence = sentences[i]
        embeddings[sentence] = embedding
    print(f"done with embeddings for {lang_code}")

# print(embeddings)

# store embeddings in json file
json_string = json.dumps(embeddings)

# Save JSON string to a file
with open("laser_embeddings_without_stopwords.json", "w") as f:
    f.write(json_string)