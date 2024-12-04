


# openai.api_key = "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"
# def get_embedding(text, model="text-embedding-ada-002"):
#     response = openai.Embedding.create(
#         input=text,
#         model=model
#     )
#     return response['data'][0]['embedding']

from openai import OpenAI
import pandas as pd
import json
import numpy as np

open_ai_api_key = "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"
client = OpenAI(api_key = open_ai_api_key)

def get_embeddings(text):
#    text = text.replace("\n", " ")
   response = client.embeddings.create(input=text, model="text-embedding-3-large")
   return [emb.embedding for emb in response.data]

# read translations.csv into a dataframe
translations = pd.read_csv('translations.csv')

# add a column called "sentence" to the dataframe that stores the 'en' column
translations['sentence'] = translations['en']
translations.set_index('sentence', inplace=True)

# key is sentence in language, value is embedding
embeddings = {}

translations.drop(columns=['ja'], inplace=True)

# get embeddings for each column other than 'sentence' column, and replace the sentence with the get_embeddings() value
for lang_code in translations.columns:
    sentences = translations[lang_code].to_list()
    vectors = get_embeddings(sentences)
    for i, embedding in enumerate(vectors):
        sentence = sentences[i]
        embeddings[sentence] = embedding
    print(f"done with embeddings for {lang_code}")

# print(embeddings)

# store embeddings in json file
json_string = json.dumps(embeddings)

# Save JSON string to a file
with open("openai_embeddings_large.json", "w") as f:
    f.write(json_string)