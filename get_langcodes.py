import os
import json
import requests
from geopy.geocoders import Nominatim
import time

# Initialize geolocator
geolocator = Nominatim(user_agent="voice-availability-map")

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

# Function to get latitude and longitude based on language code
def get_coordinates(language):
    try:
        location = geolocator.geocode(language, timeout=10)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"Error getting coordinates for {language}: {e}")
    return None, None

# Function to process language code and get coordinates
def process_language_code(lang_code):
    parts = lang_code.split('-')
    for i in range(len(parts), 0, -1):
        language = '-'.join(parts[:i])
        lat, long = get_coordinates(language)
        if lat is not None and long is not None:
            return lat, long
    return None, None

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
    # Fetch data from API
    voices = get_voices()

    # Aggregate data by language
    lang_voices_count = aggregate_voices_by_language(voices)

    # Load existing data
    geo_data, not_found = load_existing_data()

    # Get set of already processed languages
    processed_languages = {item['language_code'] for item in geo_data}

    for lang, count in lang_voices_count.items():
        if lang in processed_languages or lang in not_found:
            print(f"Skipping already processed language: {lang}")
            continue

        print(f"Looking up coordinates for language: {lang}")
        lat, long = process_language_code(lang)
        if lat is not None and long is not None:
            geo_data.append({"language_code": lang, "latitude": lat, "longitude": long})
        else:
            print(f"Could not find coordinates for language: {lang}")
            not_found.append(lang)
        
        # Save progress after each lookup
        save_progress(geo_data, not_found)

        time.sleep(1)  # To avoid overwhelming the geolocation service

    print("Geolocation lookup completed. Results saved to geo_data.json and not_found_languages.json")

if __name__ == "__main__":
    main()