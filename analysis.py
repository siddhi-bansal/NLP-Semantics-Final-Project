from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math
import json
import numpy as np

# Load the CSV files into a DataFrame
sentence_types = pd.read_csv("sentences.csv")
translations = pd.read_csv("translations.csv")

translations["sentence"] = translations["en"]
translations.drop(columns=['ja'], inplace=True)
translations.set_index('sentence', inplace=True)

with open('laser_embeddings.json', 'r') as file:
    # Load the JSON data into a dictionary
    embeddings = json.load(file)

# Extract sentences into lists based on their type
simple_sentences = sentence_types[sentence_types['Type'] == 'Simple']['Sentence'].tolist()
complex_sentences = sentence_types[sentence_types['Type'] == 'Complex']['Sentence'].tolist()
compound_sentences = sentence_types[sentence_types['Type'] == 'Compound']['Sentence'].tolist()

languages = [
    'en', 'af', 'am', 'ar', 'be', 'bn', 'bg', 'my', 'ca', 'km', 'zh', 'hr', 'cs', 'da', 'nl',
    'et', 'fi', 'fr', 'ka', 'de', 'el', 'ha', 'he', 'hi', 'hu', 'is', 'id', 'ga',
    'it', 'kk', 'ko', 'ku', 'lv', 'la', 'lt', 'mk', 'mg', 'ms', 'ml', 'mr', 'no',
    'fa', 'pl', 'pt', 'ro', 'ru', 'sr', 'sd', 'si', 'es', 'sw', 'sv', 'ta', 'te', 'th',
    'tr', 'uk', 'ur', 'uz', 'vi'
]
pairwise_combinations = list(combinations(languages, 2))

simple_cosine_avg = 0
# Compute cosine similarities for simple sentences across languages
for sentence in simple_sentences:
    sentence_cosine_avg = 0
    for l1, l2 in pairwise_combinations:
        # print("sentence", sentence)
        sentence_l1 = translations.loc[sentence, l1]
        sentence_l2 = translations.loc[sentence, l2]
        embedding_l1 = embeddings[sentence_l1]
        embedding_l2 = embeddings[sentence_l2]
        # Use numpy to calculate cos sim
        cos_sim = np.dot(embedding_l1, embedding_l2) / (np.linalg.norm(embedding_l1) * np.linalg.norm(embedding_l2))
        sentence_cosine_avg += cos_sim
    sentence_cosine_avg /= len(pairwise_combinations)
    simple_cosine_avg += sentence_cosine_avg
    print(f"done with {sentence}")
simple_cosine_avg /= len(simple_sentences)

    
compound_cosine_avg = 0
# Compute cosine similarities for compound sentences across languages
for sentence in compound_sentences:
    sentence_cosine_avg = 0
    for l1, l2 in pairwise_combinations:
        sentence_l1 = translations.loc[sentence, l1]
        sentence_l2 = translations.loc[sentence, l2]
        embedding_l1 = embeddings[sentence_l1]
        embedding_l2 = embeddings[sentence_l2]
        cos_sim = np.dot(embedding_l1, embedding_l2) / (np.linalg.norm(embedding_l1) * np.linalg.norm(embedding_l2))
        sentence_cosine_avg += cos_sim
    sentence_cosine_avg /= len(pairwise_combinations)
    compound_cosine_avg += sentence_cosine_avg
compound_cosine_avg /= len(simple_sentences)
    
complex_cosine_avg = 0
# Compute cosine similarities for simple sentences across languages
for sentence in complex_sentences:
    sentence_cosine_avg = 0
    for l1, l2 in pairwise_combinations:
        sentence_l1 = translations.loc[sentence, l1]
        sentence_l2 = translations.loc[sentence, l2]
        embedding_l1 = embeddings[sentence_l1]
        embedding_l2 = embeddings[sentence_l2]
        cos_sim = np.dot(embedding_l1, embedding_l2) / (np.linalg.norm(embedding_l1) * np.linalg.norm(embedding_l2))
        sentence_cosine_avg += cos_sim
    sentence_cosine_avg /= len(pairwise_combinations)
    complex_cosine_avg += sentence_cosine_avg
complex_cosine_avg /= len(simple_sentences)

print("Cross-lingual similarity of simple sentences", simple_cosine_avg)
print("Cross-lingual similarity of compound sentences", compound_cosine_avg)
print("Cross-lingual similarity of complex sentences", complex_cosine_avg)