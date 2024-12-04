import text2text as t2t
import pandas as pd
import json
import numpy as np

def get_embeddings(texts, code):
    embeddings = t2t.Vectorizer().transform(texts)
    return embeddings

# read translations.csv into a dataframe
translations = pd.read_csv('translations.csv')

# add a column called "sentence" to the dataframe that stores the 'en' column
translations['sentence'] = translations['en']
translations.set_index('sentence', inplace=True)

# key is sentence in language, value is embedding
embeddings = {}

translations.drop(columns=['ja'], inplace=True)

# get embeddings for each column other than 'sentence' column, and replace the sentence with the get_embeddings() value
for lang_code in translations.columns[:1]:
    sentences = translations[lang_code].to_list()
    vectors = get_embeddings(sentences, lang_code).tolist()
    for i, embedding in enumerate(vectors):
        sentence = sentences[i]
        embeddings[sentence] = embedding
    print(f"done with embeddings for {lang_code}")

# print(embeddings)

# store embeddings in json file
json_string = json.dumps(embeddings)

# Save JSON string to a file
with open("tfidf_embeddings.json", "w") as f:
    f.write(json_string)