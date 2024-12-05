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

with open('laser/laser_embeddings.json', 'r') as file:
    # Load the JSON data into a dictionary
    embeddings = json.load(file)

languages = [
    'en', 'af', 'am', 'ar', 'be', 'bn', 'bg', 'my', 'ca', 'km', 'zh', 'hr', 'cs', 'da', 'nl',
    'et', 'fi', 'fr', 'ka', 'de', 'el', 'ha', 'he', 'hi', 'hu', 'is', 'id', 'ga',
    'it', 'kk', 'ko', 'ku', 'lv', 'la', 'lt', 'mk', 'mg', 'ms', 'ml', 'mr', 'no',
    'fa', 'pl', 'pt', 'ro', 'ru', 'sr', 'sd', 'si', 'es', 'sw', 'sv', 'ta', 'te', 'th',
    'tr', 'uk', 'ur', 'uz', 'vi'
]

language_mappings = {'en': 'English','af': 'Afrikaans','am': 'Amharic','ar': 'Arabic','be': 'Belarusian','bn': 'Bengali','bg': 'Bulgarian','my': 'Burmese','ca': 'Catalan','km': 'Khmer','zh': 'Chinese','hr': 'Croatian','cs': 'Czech','da': 'Danish','nl': 'Dutch','et': 'Estonian','fi': 'Finnish','fr': 'French','ka': 'Georgian','de': 'German','el': 'Greek','ha': 'Hausa','he': 'Hebrew','hi': 'Hindi','hu': 'Hungarian','is': 'Icelandic','id': 'Indonesian','ga': 'Irish','it': 'Italian','kk': 'Kazakh','ko': 'Korean','ku': 'Kurdish','lv': 'Latvian','la': 'Latin','lt': 'Lithuanian','mk': 'Macedonian','mg': 'Malagasy','ms': 'Malay','ml': 'Malayalam','mr': 'Marathi','no': 'Norwegian','fa': 'Persian','pl': 'Polish','pt': 'Portuguese','ro': 'Romanian','ru': 'Russian','sr': 'Serbian','sd': 'Sindhi','si': 'Sinhala','es': 'Spanish','sw': 'Swahili','sv': 'Swedish','ta': 'Tamil','te': 'Telugu','th': 'Thai','tr': 'Turkish','uk': 'Ukrainian','ur': 'Urdu','uz': 'Uzbek','vi': 'Vietnamese'}

simple_sentences = sentence_types[sentence_types['Type'] == 'Simple']['Sentence'].tolist()
complex_sentences = sentence_types[sentence_types['Type'] == 'Complex']['Sentence'].tolist()
compound_sentences = sentence_types[sentence_types['Type'] == 'Compound']['Sentence'].tolist()

language_similarities = {}
sentences = compound_sentences
for l1 in languages:
    l1_similarities = {}
    for l2 in languages:
        avg_cos_sim = 0
        for sentence in sentences:
            sentence_l1 = translations.loc[sentence, l1]
            sentence_l2 = translations.loc[sentence, l2]
            embedding_l1 = embeddings[sentence_l1]
            embedding_l2 = embeddings[sentence_l2]
            # Use numpy to calculate cos sim
            cos_sim = np.dot(embedding_l1, embedding_l2) / (np.linalg.norm(embedding_l1) * np.linalg.norm(embedding_l2))
            avg_cos_sim += cos_sim
        l1_similarities[l2] = avg_cos_sim / len(sentences)
    language_similarities[l1] = l1_similarities
    print(f"done with {l1}")

# Convert the dictionary to a dataframe
language_similarities_df = pd.DataFrame(language_similarities)

language_similarities_df['lang_code'] = language_similarities_df.index
language_similarities_df['lang_code'] = language_similarities_df['lang_code'].apply(lambda x: language_mappings[x])
language_similarities_df.set_index("lang_code", inplace=True)

# Change cols, map from lang code to full language name
language_similarities_df.columns = language_similarities_df.columns.map(language_mappings)

language_similarities_df.index.set_names('Language', inplace=True)

language_similarities_df.to_csv("laser/complexity_and_similarity/compound/laser_language_similarities.csv")