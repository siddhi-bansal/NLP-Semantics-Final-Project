import pandas as pd
from google.cloud import translate_v2 as translate

def translate_text(text, target_language, translate_client):
    """Translates text into the target language."""
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

def bulk_translate_to_dataframe(texts, languages):
    """Translates an array of texts into multiple languages and returns a DataFrame."""
    # Initialize translation client
    translate_client = translate.Client.from_service_account_json('./gcp-credentials.json')
    
    # Initialize a dictionary to store the data for the DataFrame
    translation_data = {'English': texts}
    
    # Translate each sentence into each language
    for language in languages:
        translated_texts = []
        for text in texts:
            translated_text = translate_text(text, language, translate_client)
            translated_texts.append(translated_text)
        print("FINISHED LANGUAGE", language)
        translation_data[language] = translated_texts
    
    # Convert the dictionary to a DataFrame
    translation_df = pd.DataFrame(translation_data)
    return translation_df

# Array of English sentences to translate
# Populate texts with all the sentences in sentences.txt
texts = []
with open('sentences.txt', 'r') as file:
    for line in file:
        texts.append(line.strip())

# List of language codes for each target language
languages = [
    'af', 'am', 'ar', 'be', 'bn', 'bg', 'my', 'ca', 'km', 'zh', 'hr', 'cs', 'da', 'nl',
    'et', 'fi', 'fr', 'ka', 'de', 'el', 'ha', 'he', 'hi', 'hu', 'is', 'id', 'ga',
    'it', 'ja', 'kk', 'ko', 'ku', 'lv', 'la', 'lt', 'mk', 'mg', 'ms', 'ml', 'mr', 'no',
    'fa', 'pl', 'pt', 'ro', 'ru', 'sr', 'sd', 'si', 'es', 'sw', 'sv', 'ta', 'te', 'th',
    'tr', 'uk', 'ur', 'uz', 'vi'
]

# Perform the translations and get a DataFrame
translation_df = bulk_translate_to_dataframe(texts, languages)

# Display the DataFrame
translation_df.to_csv('translations.csv', index=False)