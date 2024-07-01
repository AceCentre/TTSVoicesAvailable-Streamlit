import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import requests
import matplotlib.pyplot as plt
from fuzzywuzzy import process

debug = False

# Function to fetch engines
def fetch_engines():
    response = requests.get("https://ttsvoices.acecentre.net/engines")
    engines = response.json()
    # Add "All" option
    engines = ["All"] + engines
    return engines

# Function to get voices from API
def get_voices(engine=None, lang_code=None, software=None):
    params = {}
    params['page_size'] = 0
    if engine and engine != "All":
        params['engine'] = engine.lower()  # Convert back to lowercase for the API call
    if lang_code:
        params['lang_code'] = lang_code
    if software:
        params['software'] = software
    is_development = os.getenv('DEVELOPMENT') == 'True'
    try:
        if is_development:
            response = requests.get("http://127.0.0.1:8080/voices", params=params)
        else:
            response = requests.get("https://ttsvoices.acecentre.net/voices", params=params)
        data = response.json()
        return data
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

# Determine online/offline status
online_engines = ["polly", "google", "microsoft", "elevenlabs", "witai"]
offline_engines = ["mms", "nuance-nuance", "cereproc-cereproc", "anreader-andreader", "acapela-mindexpress","microsoft-sapi","acapela-sapi","rhvoice-sapi"]

# Fetch data and prepare dataframe
voices_data = get_voices()

# Adding status to each voice
for voice in voices_data:
    engine = voice["engine"]
    if engine in online_engines:
        voice["status"] = "Online"
    else:
        voice["status"] = "Offline"

# Normalize the dataframe
df = pd.json_normalize(voices_data, 'languages', ['id', 'name', 'gender', 'engine', 'status'])

# Title and description
st.title("Text to Speech (TTS) Voice Data Explorer")
description = """
This page provides an interactive way to explore voice data from various TTS engines. 
You can filter the data by gender, status (online/offline), and search for specific languages. 
Use the map to visualize the geographical distribution of the voices. Please note though - these are approximations. Languages can be spoken anywhere in the world. 
"""
st.markdown(description)

# Add UI elements
st.sidebar.title("Filters")
show_map = st.sidebar.checkbox("Show Map", value=False)
gender_filter = st.sidebar.multiselect("Gender", options=["Male", "Female", "Unknown"], default=["Male", "Female", "Unknown"])
status_filter = st.sidebar.radio("Status", options=["All", "Online", "Offline"], index=0)

# Fetch engines and convert to title case
engines = fetch_engines()
engines_title_case = [engine.title() for engine in engines]
engine_filter = st.sidebar.selectbox("Engine", options=engines_title_case, index=0)

language_search = st.sidebar.text_input("Search Language")

if debug:
    # Debugging: Display unique engines and their counts
    st.write("Unique engines in data:", df['engine'].value_counts())

# Filter dataframe based on selections
filters_applied = False
if gender_filter:
    df = df[df['gender'].isin(gender_filter)]
    filters_applied = True
if 'status' in df.columns and status_filter != "All":
    df = df[df['status'] == status_filter]
    filters_applied = True
if engine_filter and engine_filter != "All":
    df = df[df['engine'].str.title() == engine_filter]
    filters_applied = True
if language_search:
    # Perform fuzzy matching
    all_languages = df['language'].unique()
    matches = process.extractBests(language_search, all_languages, score_cutoff=70)
    matched_languages = [match[0] for match in matches]
    df = df[df['language'].isin(matched_languages)]
    filters_applied = True

if filters_applied:
    st.markdown("<style>.markdown-text-container {display: none;}</style>", unsafe_allow_html=True)

# Calculate statistics
total_voices = len(df)
if total_voices > 0:
    online_voices = len(df[df['status'] == "Online"])
    offline_voices = len(df[df['status'] == "Offline"])
    male_voices = len(df[df['gender'] == "Male"])
    female_voices = len(df[df['gender'] == "Female"])
    unknown_gender_voices = len(df[df['gender'] == "Unknown"])

    if debug:
    # # Display statistics
        st.markdown(f"**Total Voices Found:** {total_voices}")
        st.markdown(f"**Online Voices:** {online_voices} ({online_voices / total_voices:.2%})")
        st.markdown(f"**Offline Voices:** {offline_voices} ({offline_voices / total_voices:.2%})")
        st.markdown(f"**Male Voices:** {male_voices} ({male_voices / total_voices:.2%})")
        st.markdown(f"**Female Voices:** {female_voices} ({female_voices / total_voices:.2%})")
        st.markdown(f"**Unknown Gender Voices:** {unknown_gender_voices} ({unknown_gender_voices / total_voices:.2%})")

    st.markdown(f"There are a **Total {total_voices} Voices Found:**. **Online Voices:** {online_voices} ({online_voices / total_voices:.2%}) **Offline Voices:** {offline_voices} ({offline_voices / total_voices:.2%}) **Male Voices:** {male_voices} ({male_voices / total_voices:.2%} **Female Voices:** {female_voices} ({female_voices / total_voices:.2%}) **Unknown Gender Voices:** {unknown_gender_voices} ({unknown_gender_voices / total_voices:.2%})")

else:
    st.markdown("**No voices found with the current filter selections.**")

if debug:
    # Debugging: Display first few rows of the dataframe and the engine/status counts
    st.write("Filtered dataframe:", df.head())
    st.write("Engine counts:", df['engine'].value_counts())
    st.write("Status counts:", df['status'].value_counts())

# Display dataframe without certain columns
columns_to_hide = ['latitude', 'longitude', 'id', 'name']
df_display = df.drop(columns=columns_to_hide)
st.dataframe(df_display)

# Show map if checkbox is selected
if show_map:
    lang_voices_details = aggregate_voices_by_language(voices_data)
    # Create map layer for each language
    layers = []
    for lang, details in lang_voices_details.items():
        layers.append(pdk.Layer(
            'ScatterplotLayer',
            data=pd.DataFrame({
                'lat': details['latitudes'],
                'lon': details['longitudes'],
                'count': [details['count']] * len(details['latitudes'])
            }),
            get_position='[lon, lat]',
            get_radius='[10000 * (1 + (count / 10))]',  # Adjust size based on count, smaller factor for better readability
            get_fill_color='[255, 0, 0, 100]',  # Add transparency to color
            get_line_color=[0, 0, 0, 50],  # Add slight border for better distinction
            pickable=True,
            tooltip=True
        ))

    # Set the viewport location
    view_state = pdk.ViewState(
        latitude=0,
        longitude=0,
        zoom=1,
        pitch=0,
    )

    # Render the map
    st.pydeck_chart(pdk.Deck(
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "{count} voices at this location"}
    ))
