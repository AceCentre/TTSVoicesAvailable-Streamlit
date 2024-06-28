import os
import json
import requests
import pandas as pd
from geopy.geocoders import GoogleV3
import time
import pycountry

# Initialize geolocator with GoogleV3
api_key = os.getenv('GOOGLE_GEOCODE_KEY')
if not api_key:
    raise ValueError("Google API key is not set. Please set it in the environment variable 'GOOGLE_GEOCODE_KEY'.")
geolocator = GoogleV3(api_key=api_key)

# Predefined country coordinates as a fallback
country_coordinates = {
    "Belgium": (50.8503, 4.3517),
    "United States": (37.0902, -95.7129),
    "Turkey": (38.9637, 35.2433),
    "Sweden": (60.1282, 18.6435),
    "Russia": (61.5240, 105.3188),
    "Romania": (45.9432, 24.9668),
    "Portugal": (39.3999, -8.2245),
    "Brazil": (-14.2350, -51.9253),
    "Poland": (51.9194, 19.1451),
    # Add more countries as needed
}

# Load the mmslangs_updated.tsv file
mmslangs_file_path = 'mmslangs_updated.tsv'
mmslangs_df = pd.read_csv(mmslangs_file_path, sep='\t')

# Correctly create a dictionary from the dataframe
mmslangs_dict = mmslangs_df.set_index('Iso Code').to_dict(orient='index')

def load_iso639_3_names(filename):
    iso639_3_names = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                code, name = parts[0], parts[1]
                iso639_3_names[code] = name
    return iso639_3_names

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
    response = requests.get("https://ttsvoices.acecentre.net/voices", params=params)
    return response.json()

# Function to aggregate voices by language
def aggregate_voices_by_language(data):
    lang_voices_count = {}
    for voice in data:
        for lang_code in voice['language_codes']:
            if lang_code not in lang_voices_count:
                lang_voices_count[lang_code] = 0
            lang_voices_count[lang_code] += 1
    return lang_voices_count

# Function to get latitude and longitude based on location name
def get_coordinates(location_name):
    try:
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        print(f"Error getting coordinates for {location_name}: {e}")
        return None, None

# Function to process language code and get coordinates
def process_language_code(lang_code):
    country_name = None
    region = None
    language_name = None
    lat, long = None, None

    # Check if the language code is a valid ISO 639-3 code in mmslangs_dict
    if lang_code in mmslangs_dict:
        language_name = mmslangs_dict[lang_code]['Language Name']
        country_name = mmslangs_dict[lang_code]['Country']
        region = mmslangs_dict[lang_code]['Region']
        print(f"Using language '{language_name}' for geocoding with country '{country_name}'")
        lat, long = get_coordinates(country_name)
        if lat is None and long is None and country_name in country_coordinates:
            print(f"Using fallback coordinates for {country_name}")
            lat, long = country_coordinates[country_name]
    else:
        # Try to get country name from the language tag for complex codes
        parts = lang_code.split('-')
        for i in range(len(parts), 0, -1):
            part = parts[i-1].upper()
            if len(part) == 2:
                country = pycountry.countries.get(alpha_2=part)
                if country:
                    country_name = country.name
                    break
        if country_name:
            print(f"Using country name '{country_name}' for geocoding")
            lat, long = get_coordinates(country_name)
            if lat is None and long is None and country_name in country_coordinates:
                print(f"Using fallback coordinates for {country_name}")
                lat, long = country_coordinates[country_name]
        else:
            print(f"Language code '{lang_code}' not found in mmslangs_dict or as part of a complex code")

    return lat, long, language_name, country_name, region

# Function to load existing data
def load_existing_data():
    if os.path.exists("geo_data.json"):
        with open("geo_data.json", "r") as infile:
            geo_data = json.load(infile)
    else:
        geo_data = []

    if os.path.exists("not_found_languages.json"):
        with open("not_found_languages.json", "r") as infile:
            not_found = json.load(infile)
    else:
        not_found = []

    return geo_data, not_found

# Function to save progress
def save_progress(geo_data, not_found):
    with open("geo_data.json", "w") as outfile:
        json.dump(geo_data, outfile, indent=4)

    with open("not_found_languages.json", "w") as outfile:
        json.dump(not_found, outfile, indent=4)

# Main script
def main():
    iso639_3_names = load_iso639_3_names('iso-639-3_Name_Index.tab')
    # Fetch data from API
    voices = get_voices()

    # Aggregate data by language
    lang_voices_count = aggregate_voices_by_language(voices)

    # Load existing data
    geo_data, not_found = load_existing_data()

    # Get set of already processed languages
    processed_languages = {item['language_code'] for item in geo_data}

    for lang, count in lang_voices_count.items():
        if lang in processed_languages or lang in {item['language_code'] for item in not_found}:
            print(f"Skipping already processed language: {lang}")
            continue

        print(f"Looking up coordinates for language: {lang}")
        lat, long, language_name, country, region = process_language_code(lang)
        if lat is not None and long is not None:
            geo_data.append({
                "language_code": lang,
                "latitude": lat,
                "longitude": long,
                "language": language_name,
                "country": country,
                "region": region
            })
        else:
            print(f"Could not find coordinates for language: {lang}")
            not_found.append({"language_code": lang})
        
        # Save progress after each lookup
        save_progress(geo_data, not_found)

        time.sleep(1)  # To avoid overwhelming the geolocation service

    print("Geolocation lookup completed. Results saved to geo_data.json and not_found_languages.json")

if __name__ == "__main__":
    main()
