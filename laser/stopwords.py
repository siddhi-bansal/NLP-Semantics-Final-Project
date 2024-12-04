import advertools as adv
import json
import csv

stopword_lang_code_to_language = {"en": "english","he": "hebrew","te": "telugu","fa": "persian","el": "greek","hi": "hindi","ru": "russian","ro": "romanian","no": "norwegian","fr": "french","fi": "finnish","ja": "japanese","zh": "chinese","de": "german","nl": "dutch","pt": "portuguese","hr": "croatian","ga": "irish","si": "sinhala","sv": "swedish","ur": "urdu","hu": "hungarian","da": "danish","it": "italian","ta": "tamil","ca": "catalan","tr": "turkish","ar": "arabic","id": "indonesian","bn": "bengali","vi": "vietnamese","th": "thai","kk": "kazakh","pl": "polish","es": "spanish","uk": "ukrainian"}

# create a mapping of stopword language codes to stopword languages
# access all stopwords for the languages in stopword_lang_codes, and store in a json where the key is the lang code, and the value is list of stopwords. use: adv.stopwords['english']
stopwords = {}
for lang_code in stopword_lang_code_to_language.keys():
    stopwords[lang_code] = list(adv.stopwords[stopword_lang_code_to_language[lang_code]])

print(stopwords.keys())
# stopwords has values that are not in unicode, so we can't store as string, but want to store dictionary in json. do that.
# Store the stopwords in a JSON file, ensuring non-ASCII characters are handled correctly
with open('stopwords.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Language Code', 'Stopwords'])
    
    # Write each language code and its stopwords
    for lang_code, stopword_list in stopwords.items():
        writer.writerow([lang_code, ', '.join(stopword_list)])