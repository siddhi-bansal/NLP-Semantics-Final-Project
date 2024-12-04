import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Create heatmap for language to see most similar languages
def create_heatmap_for_language(language):
    sorted_lang = language_similarities[language].sort_values(ascending=False)
    sorted_lang = sorted_lang[sorted_lang != 1]
    sorted_lang_df = sorted_lang.to_frame()
    # See heatmap for one language
    plt.figure(figsize=(20, 20))
    sns.heatmap(sorted_lang_df, fmt=".1f", cmap="coolwarm")
    plt.title(f'Heatmap of {language} Similarities', fontsize=40, pad=20)
    plt.savefig(f'heatmaps/{language.lower()}_similarity_heatmap.png', format='png')

# Create word cloud for language to see most similar languages
def create_wordcloud_for_language(language):
    sorted_lang = language_similarities[language].sort_values(ascending=False)
    sorted_lang = sorted_lang[sorted_lang != 1]
    
    wordcloud_data = sorted_lang.to_dict()
    # Create the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # No axes for the word cloud
    wordcloud.to_file(f'wordclouds/{language.lower()}_similarity_wordcloud.png')

languages = [
    'English', 'Afrikaans', 'Amharic', 'Arabic', 'Belarusian', 'Bengali', 'Bulgarian', 'Burmese', 'Catalan', 'Khmer', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'Estonian', 'Finnish', 'French', 'Georgian', 'German', 'Greek', 'Hausa', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Irish', 'Italian', 'Kazakh', 'Korean', 'Kurdish', 'Latvian', 'Latin', 'Lithuanian', 'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Marathi', 'Norwegian', 'Persian', 'Polish', 'Portuguese', 'Romanian', 'Russian', 'Serbian', 'Sindhi', 'Sinhala', 'Spanish', 'Swahili', 'Swedish', 'Tamil', 'Telugu', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese'
]

language_similarities = pd.read_csv("laser_language_similarities.csv")
language_similarities.set_index('Language', inplace=True)

sorted_language_similarities = language_similarities.copy()

# Sort by language average similarity
sorted_language_similarities['average_similarity'] = language_similarities.mean(axis=1)
sorted_language_similarities = sorted_language_similarities.sort_values(by='average_similarity', 
ascending=False)

# Sort columns too
sorted_language_similarities = sorted_language_similarities[sorted_language_similarities.index]

# See heatmap for all languages (matrix)
plt.figure(figsize=(20, 20))
sns.heatmap(sorted_language_similarities, fmt=".1f", cmap="coolwarm")
plt.title('Heatmap of Language Similarities', fontsize=20, pad=20)
plt.savefig('heatmaps/all_languages_similarity_heatmap.png', format='png')

for language in languages:
    create_heatmap_for_language(language)
    create_wordcloud_for_language(language)