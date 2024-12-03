from laserembeddings import Laser
import pandas as pd
import json

laser = Laser()

def get_embeddings(texts, code):
    embeddings = laser.embed_sentences(texts, lang=code)
    return embeddings

# read translations.csv into a dataframe
translations = pd.read_csv('translations.csv')

# add a column called "sentence" to the dataframe that stores the 'en' column
translations['sentence'] = translations['en']
translations.set_index('sentence', inplace=True)

# key is sentence in language, value is embedding
embeddings = {}

# had issues with the following language codes, so I'm blocking them
blocked_codes = ['ja']

# get embeddings for each column other than 'sentence' column, and replace the sentence with the get_embeddings() value
for code in translations.columns:
    if code not in blocked_codes:
        vector = get_embeddings(translations[code], code).tolist()
        sentence = str(translations[code]) # get sentence in language
        # print(sentence)
        embeddings[sentence] = vector

print(embeddings)

# store embeddings in json file
json_string = json.dumps(embeddings)

# Save JSON string to a file
with open("embeddings.json", "w") as f:
    f.write(json_string)
