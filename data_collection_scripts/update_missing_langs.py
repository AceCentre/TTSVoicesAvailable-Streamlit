import csv
import pycountry

# Path to the input and output CSV files
input_csv_path = 'languages_without_language.csv'
output_csv_path = 'most_likely_languages.csv'

# Load the language codes from the input CSV file
language_codes = []
with open(input_csv_path, 'r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        language_codes.append(row['language_code'])

# Helper function to identify the most likely language for a given code
def get_language_name(lang_code):
    # Try to get the language using pycountry
    lang = pycountry.languages.get(alpha_3=lang_code.lower()) or pycountry.languages.get(alpha_2=lang_code.lower())
    if lang:
        return lang.name
    
    # Handle special cases and common language code formats
    special_cases = {
        'en_US': 'English (United States)',
        'en_CA': 'English (Canada)',
        'en_GB': 'English (United Kingdom)',
        'fr_BE': 'French (Belgium)',
        'nl_BE': 'Dutch (Belgium)',
        'pt_BR': 'Portuguese (Brazil)',
        'pt_PT': 'Portuguese (Portugal)',
        'tr_TR': 'Turkish (Turkey)',
        'sv_SE': 'Swedish (Sweden)',
        'ru_RU': 'Russian (Russia)',
        'ro_RO': 'Romanian (Romania)',
        'pl_PL': 'Polish (Poland)',
        'arb': 'Standard Arabic',
        'ara': 'Arabic',
        'nan': 'Min Nan Chinese',
    }

    # Add a more comprehensive list of common language codes
    common_languages = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'pa': 'Punjabi',
        'jv': 'Javanese',
        'te': 'Telugu',
        'vi': 'Vietnamese',
        'ur': 'Urdu',
        'ta': 'Tamil',
        'mr': 'Marathi',
        'tr': 'Turkish',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'or': 'Oriya',
        'as': 'Assamese',
        'si': 'Sinhala',
        'my': 'Burmese',
        'km': 'Khmer',
        'lo': 'Lao',
        'th': 'Thai',
        'hy': 'Armenian',
        'ka': 'Georgian',
        'fa': 'Persian',
        'ps': 'Pashto',
        # Add more common languages as needed
    }

    # Return the language name from special cases or common languages
    return special_cases.get(lang_code) or common_languages.get(lang_code[:2], 'Unknown')

# Prepare the data for CSV output
output_data = []
for lang_code in language_codes:
    language_name = get_language_name(lang_code)
    output_data.append({'language_code': lang_code, 'language_name': language_name})

# Write the output to a CSV file
with open(output_csv_path, 'w', newline='') as csvfile:
    fieldnames = ['language_code', 'language_name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(output_data)

print(f"Most likely languages have been saved to {output_csv_path}")
