import pycountry
import langcodes
import os
from geopy.geocoders import GoogleV3
import requests
import json
import time
import csv

## NB We dont really use this hacky GPT technique anymore. Ive left it in for history
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
                {"role": "system", "content": "You are a helpful assistant that gives me the language spoken given a code. Don't provide any other text. If you don't know just say unknown. Codes may be short language codes eg kur or locale codes like en-GB or en-GB-WLS"},
                {"role": "user", "content": input_string},
            ]
        )
        corrected_text = response.choices[0].message.content.strip()
        return corrected_text
    except Exception as e:
        logger.error(f"Error in get_language_gpt: {e}")
        return "unknown"

def get_country_gpt(azure_openai_client, input_string):
    try:
        response = azure_openai_client.chat.completions.create(
            model="correctasentence", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant that gives me the most likely country and if possible a region when provided a badly formatted country code. Just provide the country name to your best guess. You are an API"},
                {"role": "user", "content": f"What is the most likely country for the language {input_string}?"},
            ]
        )
        print(f"Response: {response}")
        corrected_text = response.choices[0].message.content.strip()
        print(f"Corrected text: {corrected_text}")
        if corrected_text.lower() == 'unknown':
            return "unknown", "unknown"
        return "unknown", corrected_text  # Return as a tuple with region as "unknown"
    except Exception as e:
        logger.error(f"Error in get_country_gpt: {e}")
        return "unknown", "unknown"


# Function to get voices from API
def get_voices(engine=None, lang_code=None, software=None):
    params = {}
    params['page_size'] = 0
    if engine:
        params['engine'] = engine
    if lang_code:
        params['lang_code'] = lang_code
    if software:
        params['software'] = software
    #response = requests.get("https://ttsvoices.acecentre.net/voices", params=params)
    response = requests.get("http://0.0.0.0:8080/voices", params=params)
    return response.json()

# Function to load existing data
def load_existing_data():
    geo_data = []
    not_found = []

    if os.path.exists("geo_data.json"):
        try:
            with open("geo_data.json", "r") as infile:
                geo_data = json.load(infile)
        except json.JSONDecodeError:
            print("geo_data.json is empty or corrupted. Initializing with an empty list.")

    if os.path.exists("not_found_languages.json"):
        try:
            with open("not_found_languages.json", "r") as infile:
                not_found = json.load(infile)
        except json.JSONDecodeError:
            print("not_found_languages.json is empty or corrupted. Initializing with an empty list.")

    return geo_data, not_found


# Function to save progress
def save_progress(geo_data, not_found):
    with open("geo-data.json", "w") as outfile:
        json.dump(geo_data, outfile, indent=4)

    with open("not_found_languages.json", "w") as outfile:
        json.dump(not_found, outfile, indent=4)


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
    else:
        # Get the country code and region if available
        if len(parts) > 1:
            if len(parts[1]) == 2:
                country = pycountry.countries.get(alpha_2=parts[1])
                if country:
                    country_code = country.alpha_2

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
                    if country:
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

# Initialize geolocator with GoogleV3
api_key = os.getenv('GOOGLE_GEOCODE_KEY')
if not api_key:
    raise ValueError("Google API key is not set. Please set it in the environment variable 'GOOGLE_GEOCODE_KEY'.")
geolocator = GoogleV3(api_key=api_key)

# Function to get latitude and longitude based on location name
def get_coordinates(location_name):
    #return 0,0
    try:
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error getting coordinates for {location_name}: {e}")
        return None, None

# Function to aggregate voices by language
def aggregate_voices_by_language(data):
    lang_voices_count = {}
    for voice in data:
        for voice in data:
            for language in voice['languages']:
                try:
                    lang_code = language['language_code']
                    if lang_code not in lang_voices_count:
                        lang_voices_count[lang_code] = 0
                    lang_voices_count[lang_code] += 1
                except KeyError as e:
                    print(f"KeyError: {e}")
                    print(f"Voice: {voice}")
                    print(f"Language: {language}") 
                except Exception as e:
                    print(f"Error: {e}")
    return lang_voices_count


# Main script
def main():
    # Fetch data from API
    voices = get_voices()
    # Load existing data
    geo_data, not_found = load_existing_data()
    # Get set of already processed languages
    processed_languages = {item['language_code'] for item in geo_data}
    # Get unique languages and count of voices for each language
    lang_voices_count = aggregate_voices_by_language(voices)

    for lang, count in lang_voices_count.items():
        if lang in processed_languages or lang in {item['language_id'] for item in not_found}:
            print(f"Skipping already processed language: {lang}")
            continue

        print(f"Looking up coordinates for language: {lang}")
        iso639_3_info = locale_to_iso639_info(lang, best_guess=True)
        iso639_3 = iso639_3_info['iso639_3']
        country_code = iso639_3_info['country_code']
        region_code = iso639_3_info['region_code']
        full_english_name = iso639_3_info['full_english_name']
        geolocatable_string = iso639_3_info['geolocatable_string']
        written_script = iso639_3_info['written_script']
        lat, long = None, None
        print(f"ISO 639-3 code: {iso639_3_info}")

        if full_english_name == 'unknown':
            print(f"!!! Skipping language with unknown ISO 639-3 code: {lang}")
            for voice in voices:
                if lang in voice['languages']:
                    print(f"Voice: {voice}")
                    break
            continue
        if geolocatable_string != 'unknown':
            print(f"Geolocatable string: {geolocatable_string}")
            lat, long = get_coordinates(geolocatable_string)
        if lat is not None and long is not None:
            geo_data.append({
                "language_id": lang,
                "language_code": iso639_3,
                "latitude": lat,
                "longitude": long,
                "language": full_english_name,
                "country": country_code,
                "region": region_code,
                "written_script": written_script
            })
        else:
            print(f"Could not find coordinates for language: {lang}")
            not_found.append({"language_id": lang})
        
        # Save progress after each lookup
        save_progress(geo_data, not_found)

        time.sleep(1)  # To avoid overwhelming the geolocation service

    print("Geolocation lookup completed. Results saved to geo_data.json and not_found_languages.json")

if __name__ == "__main__":
    main()
