import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import requests

def fetch_engines():
    response = requests.get("https://ttsvoices.acecentre.net/engines")
    engines = response.json()
    # Add "All" and "Other" options
    engines = ["All"] + engines
    return engines


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
    is_development = os.getenv('DEVELOPMENT') == 'True'
    try:
        if is_development:
            #response = requests.get("http://127.0.0.1:8000/voices", params=params)
            response = requests.get("https://ttsvoices.acecentre.net/voices", params=params)
        else:
            response = requests.get("https://ttsvoices.acecentre.net/voices", params=params)
        data = response.json()
        return data
        print(data)
    except Exception as e:
        st.error(f"Error retrieving voices: {e}")
        return []

# Function to aggregate voices by language and collect coordinates
def aggregate_voices_by_language(data):
    lang_voices_details = {}
    for voice in data:
        for language in voice['languages']:
            lang_code = language['language_code']
            if lang_code not in lang_voices_details:
                lang_voices_details[lang_code] = {'count': 0, 'latitudes': [], 'longitudes': []}
            lang_voices_details[lang_code]['count'] += 1
            if language['latitude'] != "0.0" and language['longitude'] != "0.0":  # Check if the location is valid
                lang_voices_details[lang_code]['latitudes'].append(float(language['latitude']))
                lang_voices_details[lang_code]['longitudes'].append(float(language['longitude']))
    return lang_voices_details

# Get voices from the API
voices = get_voices()

# Prepare data for map
map_data = []
for voice in voices:
    for language in voice['languages']:
        if language['latitude'] != "0.0" and language['longitude'] != "0.0":  # Check if the location is valid
            map_data.append({
                'Language': language['language'],
                'Voices': 1,
                'Latitude': float(language['latitude']),
                'Longitude': float(language['longitude'])
            })

# Convert map data to DataFrame
df = pd.DataFrame(map_data)

# Aggregate data by language and coordinates
aggregated_data = df.groupby(['Language', 'Latitude', 'Longitude']).size().reset_index(name='Voices')

# Debugging: Print the DataFrame to ensure it includes all expected voices
#st.write("### Aggregated Voices Data", aggregated_data)

# Adjust the initial PyDeck layer radius
scaling_factor = 10000  # Increase this value to make the dots larger
max_radius = 500000  # Increase this value to allow for larger maximum dot sizes

# Calculate radius based on the count of voices for each language
aggregated_data['Radius'] = aggregated_data['Voices'].apply(lambda x: min(x * scaling_factor, max_radius))  # Adjust scaling as needed

# Debugging: Print the DataFrame with the calculated radius
#st.write("### Aggregated Voices Data with Radius", aggregated_data)

layer = pdk.Layer(
    'ScatterplotLayer',
    data=aggregated_data,
    get_position='[Longitude, Latitude]',
    get_radius='Radius',  # Use the pre-calculated radius column
    get_fill_color='[200, 30, 0, 80]',
    pickable=True
)

tool_tip = {"html": "Language: {Language}<br/>Voices: {Voices}", "style": {"color": "white"}}
view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1)  # Adjust the initial view state as needed
map = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tool_tip)
st.pydeck_chart(map)

# Filter Form
st.write("## Filter Voices")
engines_list = fetch_engines()
engine = st.selectbox("Select Engine", options=engines_list)

# Collect all unique languages for the dropdown
all_languages = set()
for voice in voices:
    all_languages.update([lang['language'] for lang in voice['languages']])
# Convert the set to a sorted list to have a consistent order
language_options = sorted(list(all_languages))
# Use these language names as options in a Streamlit selectbox
selected_language = st.selectbox("Select a language:", ["All"] + language_options)

# Filter voices based on engine and selected language
filtered_voices = [voice for voice in voices if (voice["engine"] == engine or engine == "All") and 
                   any(lang["language"] == selected_language or selected_language == "All" for lang in voice['languages'])]

# Transform 'languages' in each voice dictionary to a string representation before creating the DataFrame
for voice in filtered_voices:
    voice['languages'] = ', '.join([lang['language'] for lang in voice['languages']])

# Create the DataFrame from the updated filtered_voices list
filtered_voices_df = pd.DataFrame(filtered_voices)

# Display the number of voices found and the DataFrame
st.write(f"### {len(filtered_voices_df)} Voices Found")
st.dataframe(filtered_voices_df.style.set_properties(**{'width': '800px', 'height': '400px'}))
