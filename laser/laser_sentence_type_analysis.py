from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math
import json
import numpy as np
import matplotlib.pyplot as plt

def create_boxplots(simple_cosines, compound_cosines, complex_cosines):
    data = [simple_cosines, compound_cosines, complex_cosines]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, patch_artist=True, notch=True, showmeans=True)

    # Add titles and labels
    plt.title("Spread of Cross-Lingual Cosine Similarities for Sentences Across Sentence Types", fontsize=16)
    plt.xticks([1, 2, 3], ["Simple", "Complex", "Compound"], fontsize=12)
    plt.ylabel("Average Cosine Similarity", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot as a PNG file
    plt.savefig("laser_sentence_types_boxplot.png", format="png")

# Load the CSV files into a DataFrame
sentence_types = pd.read_csv("sentences.csv")
translations = pd.read_csv("translations.csv")

translations["sentence"] = translations["en"]
translations.drop(columns=['ja'], inplace=True)
translations.set_index('sentence', inplace=True)

with open('laser_embeddings.json', 'r') as file:
    # Load the JSON data into a dictionary
    embeddings = json.load(file)

languages = [
    'en', 'af', 'am', 'ar', 'be', 'bn', 'bg', 'my', 'ca', 'km', 'zh', 'hr', 'cs', 'da', 'nl',
    'et', 'fi', 'fr', 'ka', 'de', 'el', 'ha', 'he', 'hi', 'hu', 'is', 'id', 'ga',
    'it', 'kk', 'ko', 'ku', 'lv', 'la', 'lt', 'mk', 'mg', 'ms', 'ml', 'mr', 'no',
    'fa', 'pl', 'pt', 'ro', 'ru', 'sr', 'sd', 'si', 'es', 'sw', 'sv', 'ta', 'te', 'th',
    'tr', 'uk', 'ur', 'uz', 'vi'
]

# Extract sentences into lists based on their type
simple_sentences = sentence_types[sentence_types['Type'] == 'Simple']['Sentence'].tolist()
complex_sentences = sentence_types[sentence_types['Type'] == 'Complex']['Sentence'].tolist()
compound_sentences = sentence_types[sentence_types['Type'] == 'Compound']['Sentence'].tolist()
pairwise_combinations = list(combinations(languages, 2))

simple_cosine_avg = 0
simple_sentence_cosine_avgs = []
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
    simple_sentence_cosine_avgs.append(sentence_cosine_avg)
    simple_cosine_avg += sentence_cosine_avg
    print(f"done with simple sentence: {sentence}")
simple_cosine_avg /= len(simple_sentences)

    
compound_cosine_avg = 0
compound_sentence_cosine_avgs = []
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
    compound_sentence_cosine_avgs.append(sentence_cosine_avg)
    compound_cosine_avg += sentence_cosine_avg
    print(f"done with compound sentence: {sentence}")
compound_cosine_avg /= len(simple_sentences)
    
complex_cosine_avg = 0
complex_sentence_cosine_avgs = []
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
    complex_sentence_cosine_avgs.append(sentence_cosine_avg)
    complex_cosine_avg += sentence_cosine_avg
    print(f"done with complex sentence: {sentence}")
complex_cosine_avg /= len(simple_sentences)

create_boxplots(simple_sentence_cosine_avgs, compound_sentence_cosine_avgs, complex_sentence_cosine_avgs)
print("Cross-lingual similarity of simple sentences for Laser Embeddings", simple_cosine_avg)
print("Cross-lingual similarity of compound sentences for Laser embeddings", compound_cosine_avg)
print("Cross-lingual similarity of complex sentences for Laser Embeddings", complex_cosine_avg)