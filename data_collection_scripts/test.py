import pycountry
import langcodes
import os
import csv
from openai import AzureOpenAI

def setup_openAI():
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),  
        api_version="2023-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    return client

azure_openai_client = setup_openAI()

def get_language_gpt(azure_openai_client, input_string):
    try:
        response = azure_openai_client.chat.completions.create(
            model="correctasentence", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant that gives me the language spoken given a code. Don't provide any other text. If you don't know just say Unknown. Codes may be short language codes eg kur or locale codes like en-GB or en-GB-WLS"},
                {"role": "user", "content": input_string},
            ]
        )
        corrected_text = response.choices[0].message.content.strip()
        return corrected_text
    except Exception as e:
        print(f"Error in get_language_gpt: {e}")
        return "Unknown"

def get_country_gpt(azure_openai_client, input_string):
    try:
        response = azure_openai_client.chat.completions.create(
            model="correctasentence", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant that gives me the most likely country and if possible a region when provided a badly formatted country code. Just provide the country name to your best guess. You are an API"},
                {"role": "user", "content": f"What is the most likely country for the language {input_string}?"},
            ]
        )
        corrected_text = response.choices[0].message.content.strip()
        if corrected_text.lower() == 'unknown':
            return "Unknown", "Unknown"
        return "Unknown", corrected_text  # Return as a tuple with region as "Unknown"
    except Exception as e:
        print(f"Error in get_country_gpt: {e}")
        return "Unknown", "Unknown"

region_names = {
    'WLS': 'Wales',
    'SCT': 'Scotland'
    # Add more regions as needed
}

custom_code_mapping = {
    'ARG': 'arb-146',
    'ARW': 'ar-WW',
    'BAE': 'eu-ES',
    'BEI': 'bn-IN',
    'BGB': 'bg-BG',
    'BHI': 'bho-IN',
    'CAE': 'ca-ES',
    'CAH': 'zh-HK',
    'CZC': 'cs-CZ',
    'DAD': 'da-DK',
    'DUB': 'nl-BE',
    'DUN': 'nl-NL',
    'ENA': 'en-AU',
    'ENE': 'en-IE',
    'ENG': 'en-GB',
    'ENI': 'en-IN',
    'ENS': 'en-GB-SCT',
    'ENU': 'en-US',
    'ENZ': 'en-ZA',
    'FAI': 'fa-IR',
    'FIF': 'fi-FI',
    'FRB': 'fr-BE',
    'FRC': 'fr-CA',
    'FRF': 'fr-FR',
    'GED': 'de-DE',
    'GLE': 'gl-ES',
    'GRG': 'el-GR',
    'HEI': 'he-IL',
    'HII': 'hi-IN',
    'HRH': 'hr-HR',
    'HUH': 'hu-HU',
    'IDI': 'id-ID',
    'ITI': 'it-IT',
    'JPJ': 'ja-JP',
    'KAI': 'kn-IN',
    'KOK': 'ko-KR',
    'MAI': 'mr-IN',
    'MNC': 'zh-CN',
    'MNT': 'zh-TW',
    'MSM': 'ms-MY',
    'NON': 'no-NO',
    'PLP': 'pl-PL',
    'PTB': 'pt-BR',
    'PTP': 'pt-PT',
    'ROR': 'ro-RO',
    'RUR': 'ru-RU',
    'SKS': 'sk-SK',
    'SPA': 'es-AR',
    'SPC': 'es-CO',
    'SPE': 'es-ES',
    'SPL': 'es-CL',
    'SPM': 'es-MX',
    'SWS': 'sv-SE',
    'TAI': 'ta-IN',
    'TEI': 'te-IN',
    'THT': 'th-TH',
    'TRT': 'tr-TR',
    'UKU': 'uk-UA',
    'VAE': 'ca-ES-valencia',
    'VIV': 'vi-VN',
    'ac-ac': 'ar-SA',
    'cn': 'zh-CN',
    'cz': 'cs-CZ',
    'dk': 'da-DK',
    'au': 'en-AU',
    'in': 'hi-IN',
    'gb-sct': 'en-GB-SCT',
    'gb': 'en-GB',
    'us': 'en-US',
    'ac-qc': 'fr-CA',
    'gr': 'el-GR',
    'jp': 'ja-JP',
    'ac-si': 'se-NO'
}

# Custom country and region mapping from TSV file
custom_country_region_mapping = {}

# Read the TSV file and populate the mapping
with open('mmslangs_updated.tsv', newline='', encoding='utf-8') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        iso_code = row['Iso Code'].strip()
        country = row['Country'].strip()
        region = row['Region'].strip()
        custom_country_region_mapping[iso_code] = {'country': country, 'region': region}

def get_country_code(country_name):
    try:
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_2
        else:
            return "unknown"
    except LookupError:
        return "unknown"

def locale_to_iso639_info(locale_code, best_guess=False):
    """
    Convert a locale code to an ISO 639-3 language code, region code, and country code.

    Parameters:
    locale_code (str): The locale code (e.g., 'en-US', 'en-GB', 'en-GB-WLS', 'en_GB', 'eng').
    best_guess (bool): Flag to enable best guess for country code if not provided.

    Returns:
    dict: A dictionary with the ISO 639-3 language code, region code, country code, full English name,
          and a geolocatable string.
    """
    # Normalize locale code by replacing underscores with hyphens
    normalized_code = locale_code.replace('_', '-')
    parts = normalized_code.split('-')
    
    # Default values
    iso639_3 = 'unknown'
    country_code = 'unknown'
    region_code = 'unknown'
    full_english_name = 'unknown'
    geolocatable_string = 'unknown'
    written_script = 'unknown'
    
    # Check custom code mapping first
    if locale_code in custom_code_mapping:
        locale_code = custom_code_mapping[locale_code]
        parts = locale_code.split('-')

    # Determine if the language part is ISO 639-1 or ISO 639-3
    language_code = parts[0]
    if len(language_code) == 2:
        language = pycountry.languages.get(alpha_2=language_code)
    elif len(language_code) == 3:
        language = pycountry.languages.get(alpha_3=language_code)
    else:
        language = None
    
    if language:
        iso639_3 = language.alpha_3
        full_english_name = language.name

    # Check custom mapping for country and region
    if language_code in custom_country_region_mapping:
        custom_info = custom_country_region_mapping[language_code]
        country_name = custom_info['country']
        region_name = custom_info['region']
        country_code = get_country_code(country_name)
        geolocatable_string = f"{country_name}, {region_name}"
        if country_name and country_code != language.alpha_2:
            full_english_name += f", {country_name}"
    else:
        # Get the country code and region if available
        if len(parts) > 1:
            if len(parts[1]) == 2:
                country = pycountry.countries.get(alpha_2=parts[1])
                if country:
                    country_code = country.alpha_2
                    if country_code != language.alpha_2:
                        full_english_name += f", {country.name}"

        if len(parts) > 2:
            # Look up the full name for the region if it exists
            if parts[1] == 'script':
                written_script = parts[2]
                if written_script == 'cyrillic':
                    written_script = 'Cyrl'
                elif written_script == 'latin':
                    written_script = 'Latn'
                elif written_script == 'arabic':
                    written_script = 'Arab'
            else:
                region_code = parts[2]
                region_name = region_names.get(region_code, region_code)
                full_english_name += f", {region_name}"

        # If best guess is enabled and country code is unknown
        if best_guess and country_code == 'unknown' and language:
            # Use langcodes library to guess the most likely country for the given language
            try:
                expanded_tag = langcodes.Language.get(language_code).to_tag()
                likely_tag = langcodes.get(expanded_tag)
                likely_country = likely_tag.maximize().territory
                if likely_country:
                    country_code = likely_country
                    country = pycountry.countries.get(alpha_2=likely_country)
                    if country and country_code != language.alpha_2:
                        full_english_name = f"{full_english_name}, {country.name}"
                        geolocatable_string = f"{region_name}, {country.name}" if region_code != 'unknown' else country.name
                else:
                    country_code = 'ZZ'  # Unknown or unspecified country code
            except:
                country_code = 'unknown'
    
    # Finally get script if unknown
    if written_script == 'unknown':
        try:
            language_info = langcodes.Language.get(normalized_code).maximize()
            written_script = language_info.script
        except:
            written_script = 'unknown'
    
    # Fallback to GPT for country code if still unknown
    if country_code == 'ZZ' and full_english_name != 'unknown':
        region_name, country_name = get_country_gpt(azure_openai_client, full_english_name)
        if country_name != 'Unknown':
            country_code = get_country_code(country_name)
            geolocatable_string = country_name
            if country_name:
                full_english_name += f", {country_name}"
    
    # Avoid appending country names to languages that don't need it
    if country_code == 'unknown' or full_english_name == language.name:
        full_english_name = language.name if language else 'unknown'
    
    # Ensure geolocatable_string is set correctly if possible
    if geolocatable_string == 'unknown' and country_code != 'unknown':
        country = pycountry.countries.get(alpha_2=country_code)
        if country:
            geolocatable_string = country.name

    return {
        'iso639_3': iso639_3,
        'country_code': country_code,
        'region_code': region_code,
        'full_english_name': full_english_name,
        'geolocatable_string': geolocatable_string,
        'written_script': written_script
    }

# Example use of the function
language_codes = [
    'en-gb-sct', 'en-us', 'eng', 'he-is', 'ENS', 'ENZ', 'FRB', 'HRH', 'JPJ', 'PJ', 'MNT',
    'PLP', 'PTB', 'RUR', 'TAI', 'ac-ac', 'cn', 'cz', 'dk', 'au', 'in', 'gb-sct',
    'gb', 'us', 'ac-qc', 'gr', 'jp', 'abi'
]

for code in language_codes:
    data = locale_to_iso639_info(code, best_guess=True)
    print(f"Locale code: {code}:")
    print(data)
    print("\n")
