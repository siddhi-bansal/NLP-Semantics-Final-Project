from itertools import combinations
import json
from wordfreq import word_frequency
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# The following function calculates the average word frequency of a sentence in a given language
def avg_word_freq(sentence, lang):
    words = sentence.split()
    freqs = [word_frequency(word, lang) for word in words]
    return sum(freqs) / len(freqs)

# The following function calculates the minimum word frequency of a sentence in a given language
def min_word_freq(sentence, lang):
    words = sentence.split()
    freqs = [word_frequency(word, lang) for word in words]
    return min(freqs)

all_languages = [
    'en', 'af', 'am', 'ar', 'be', 'bn', 'bg', 'my', 'ca', 'km', 'zh', 'hr', 'cs', 'da', 'nl',
    'et', 'fi', 'fr', 'ka', 'de', 'el', 'ha', 'he', 'hi', 'hu', 'is', 'id', 'ga',
    'it', 'kk', 'ko', 'ku', 'lv', 'la', 'lt', 'mk', 'mg', 'ms', 'ml', 'mr', 'no',
    'fa', 'pl', 'pt', 'ro', 'ru', 'sr', 'sd', 'si', 'es', 'sw', 'sv', 'ta', 'te', 'th',
    'tr', 'uk', 'ur', 'uz', 'vi'
]

word_freq_lang_codes = [
    "ar", "bn", "bg", "ca", "cs", "da", "nl", "en", "fi", 
    "fr", "de", "el", "he", "hi", "hu", "is", "id", "it", "lv", 
    "lt", "mk", "ms", "nb", "fa", "pl", "pt", "ro", "ru", "sk", "sl",
    "es", "sv", "fil", "ta", "tr", "uk", "ur", "vi"
]

# language_codes are interscetionn of word_freq_lang_codes and all_languages
# language_codes = list(set(word_freq_lang_codes).intersection(all_languages))
language_codes = ['is', 'ru', 'ro', 'bn', 'nl', 'fa', 'uk', 'bg', 'en', 'de', 'ms', 'fi', 'he', 'hu', 'pl', 'tr', 'es', 'da', 'el', 'cs', 'ca', 'mk', 'fr', 'id', 'lv', 'pt', 'sv', 'ur', 'vi', 'ta', 'it', 'ar', 'lt', 'hi']
print("language_codes", language_codes)
translations = pd.read_csv("../translations.csv")

translations["sentence"] = translations["en"]
# only have columns for languages in language_codes
translations = translations[language_codes + ["sentence"]]
translations.set_index('sentence', inplace=True)

# go through each sentence and lang code, and get the average word frequency for that sentence in that language
# and store it in a dictionary where the key is the sentence and the value is the average word frequency
avg_word_freqs = {}
for lang_code in language_codes:
    sentences = translations[lang_code].to_list()
    for sentence in sentences:
        avg_freq = avg_word_freq(sentence, lang_code)
        avg_word_freqs[sentence] = avg_freq
    print(f"done with {lang_code}")


# do the above for minimum word frequency
min_word_freqs = {}
for lang_code in language_codes:
    sentences = translations[lang_code].to_list()
    for sentence in sentences:
        min_freq = min_word_freq(sentence, lang_code)
        min_word_freqs[sentence] = min_freq
    print(f"done with {lang_code}")

sentence_types = pd.read_csv("../sentences.csv")
simple_sentences = sentence_types[sentence_types['Type'] == 'Simple']['Sentence'].tolist()
complex_sentences = sentence_types[sentence_types['Type'] == 'Complex']['Sentence'].tolist()
compound_sentences = sentence_types[sentence_types['Type'] == 'Compound']['Sentence'].tolist()

simple_avg_word_freqs = [avg_word_freqs[sentence] for sentence in simple_sentences]
simple_min_word_freqs = [min_word_freqs[sentence] for sentence in simple_sentences]

compound_avg_word_freqs = [avg_word_freqs[sentence] for sentence in compound_sentences]
compound_min_word_freqs = [min_word_freqs[sentence] for sentence in compound_sentences]

complex_avg_word_freqs = [avg_word_freqs[sentence] for sentence in complex_sentences]
complex_min_word_freqs = [min_word_freqs[sentence] for sentence in complex_sentences]

# i want to visualize the average word frequency of simple, complex, and compound sentences with three boxplots
# one for each type of sentence
plt.boxplot([simple_avg_word_freqs, complex_avg_word_freqs, compound_avg_word_freqs], labels=["Simple", "Complex", "Compound"])
plt.title("Average Word Frequency of Simple, Complex, and Compound Sentences")
# plt.show()

# save to file
plt.savefig("avg_word_freqs.png")

# i want to visualize the minimum word frequency of simple, complex, and compound sentences with three boxplots, one for each type of sentence
plt.boxplot([simple_min_word_freqs, complex_min_word_freqs, compound_min_word_freqs], labels=["Simple", "Complex", "Compound"])
plt.title("Minimum Word Frequency of Simple, Complex, and Compound Sentences")
# plt.show()

# save to file
plt.savefig("min_word_freqs.png")

# get 25th percentile of a list
rare_threshold = (np.percentile(simple_min_word_freqs, 25) + np.percentile(compound_min_word_freqs, 25) + np.percentile(complex_min_word_freqs, 25)) / 3

def classify_sentence_rarity(sentence, lang_code, rare_threshold):    
    return min_word_freq(sentence, lang_code) < rare_threshold

# df: translations

# sentence language rarity

sentence_rarity = pd.DataFrame(columns=["sentence", "language", "rarity"])
num_trues = 0
num_falses = 0
avg_word_freqs = {}
# replace ['en] with language_codes if required
for lang_code in ['en']:
    sentences = translations[lang_code].to_list()
    for sentence in sentences:
        rarity = classify_sentence_rarity(sentence, lang_code, rare_threshold)
        if rarity:
            num_trues += 1
        else:
            num_falses += 1
        # add row to sentence_rarity dataframe with sentence, lang_code, and rarity
        sentence_rarity = sentence_rarity._append({"sentence": sentence, "language": lang_code, "rarity": rarity}, ignore_index=True)
    print(f"done with {lang_code}")

print(f"num_trues: {num_trues}, num_falses: {num_falses}")


# export sentence_rarity to csv
sentence_rarity.to_csv("sentence_rarity.csv", index=False)


sentence_rarity = pd.read_csv("sentence_rarity.csv")

with open('laser_embeddings.json', 'r') as file:
    # Load the JSON data into a dictionary
    embeddings = json.load(file)

# Extract sentences into lists based on their type
true_rarity_sentences = sentence_rarity[sentence_rarity['rarity'] == True]['sentence'].tolist()
false_rarity_sentences = sentence_rarity[sentence_rarity['rarity'] == False]['sentence'].tolist()


# i want to get cosine similarities for each sentence in true rarity sentences with pairwise combinaitons of each other (within the sentences) and store the average cosine similarity for each sentence in true rarity sentences

# get pairwise combinations of sentences with each other, and get the average cosine similarity for each sentence pair, but only for true sentences

def cosine_similarity(emb1, emb2):
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return cos_sim

true_sentence_cosine_avgs = []
for sentence in true_rarity_sentences:
    emb1 = embeddings[sentence]
    avg_cosine_sim = 0
    for other_sentence in true_rarity_sentences:
        if sentence != other_sentence:
            emb2 = embeddings[other_sentence]
            avg_cosine_sim += cosine_similarity(emb1, emb2)
    avg_cosine_sim /= (len(true_rarity_sentences) - 1)
    true_sentence_cosine_avgs.append(avg_cosine_sim)

# do the above for false rarity sentences
false_sentence_cosine_avgs = []
for sentence in false_rarity_sentences:
    emb1 = embeddings[sentence]
    avg_cosine_sim = 0
    for other_sentence in false_rarity_sentences:
        if sentence != other_sentence:
            emb2 = embeddings[other_sentence]
            avg_cosine_sim += cosine_similarity(emb1, emb2)
    avg_cosine_sim /= (len(false_rarity_sentences) - 1)
    false_sentence_cosine_avgs.append(avg_cosine_sim)

def create_boxplots(true_cosines, false_cosines):
    data = [true_cosines, false_cosines]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, patch_artist=True, notch=True, showmeans=True)

    # Add titles and labels
    plt.title("Spread of Cross-Lingual Cosine Similarities for Sentences Across Sentence Rarity for English", fontsize=16)
    plt.xticks([1, 2], ["True", "False"], fontsize=12)
    plt.ylabel("Average Cosine Similarity", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot as a PNG file
    plt.savefig("laser_sentence_rarity_boxplot.png", format="png")

create_boxplots(true_sentence_cosine_avgs, false_sentence_cosine_avgs)
# print out five number summary for both lis†s
print("True sentence cosine similarities 5 number summary", np.percentile(true_sentence_cosine_avgs, [0, 25, 50, 75, 100]))
print("False sentence cosine similarities 5 number summary", np.percentile(false_sentence_cosine_avgs, [0, 25, 50, 75, 100]))
print("Cross-lingual similarity of true rarity sentences for Laser Embeddings", true_cosine_avg)
print("Cross-lingual similarity of false rarity sentences for Laser embeddings", false_cosine_avg)
