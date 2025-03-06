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
    # Add filter options with exact case matching
    engines = [
        "All",
        "All - Except MMS",  # Match the case that Streamlit is showing
        "All - Except eSpeak",
        "All - Except MMS & eSpeak"
    ] + engines
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
            #response = requests.get("http://127.0.0.1:8080/voices", params=params)
            response = requests.get("https://ttsvoices.acecentre.net/voices", params=params)
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
    for index, row in data.iterrows():
        lang_code = row['language_code']
        if lang_code not in lang_voices_details:
            lang_voices_details[lang_code] = {'count': 0, 'latitudes': [], 'longitudes': []}
        lang_voices_details[lang_code]['count'] += 1
        if row['latitude'] != "0.0" and row['longitude'] != "0.0":  # Check if the location is valid
            lang_voices_details[lang_code]['latitudes'].append(float(row['latitude']))
            lang_voices_details[lang_code]['longitudes'].append(float(row['longitude']))
    return lang_voices_details

# Fetch data and prepare dataframe
voices_data = get_voices()
# print("\n=== Initial Data Load ===")
# print(f"Raw data length: {len(voices_data)}")

# Normalize the dataframe and set status based on is_offline
df = pd.json_normalize(voices_data, 'languages', ['id', 'name', 'gender', 'engine'])
df['status'] = pd.json_normalize(voices_data, 'languages', ['is_offline'])['is_offline'].apply(lambda x: 'Offline' if x else 'Online')

# print(f"DataFrame length after normalize: {len(df)}")
# print("\nInitial engine distribution:")
# print(df['engine'].value_counts())

# Map gender values to standardized values
df['gender'] = df['gender'].str.lower().replace({
    'm': 'Male', 'male': 'Male', 'masc': 'Male', 'masculine': 'Male',
    'f': 'Female', 'female': 'Female', 'fem': 'Female', 'feminine': 'Female'
})
df['gender'] = df['gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'Unknown')

# Title and description
st.title("Text to Speech (TTS) Voice Data Explorer")
description = """
This page provides an interactive way to explore voice data from various TTS engines. 
You can filter the data by gender, status (online/offline), and search for specific languages. 
Use the map to visualize the geographical distribution of the voices. Please note though - these are approximations. Languages can be spoken anywhere in the world. 
"""
st.markdown(description)

# Toggleable long description
show_long_text = st.checkbox("Show More Information")
if show_long_text:
    long_text = """
    If the voice is 'online' you can use it with a bridging software on Windows using [AAC Speak Helper](https://docs.acecentre.org.uk/products/v/aac-speak-helper-tool). This will work with Windows AAC software such as [The Grid 3 on Windows](http://thinksmartbox.com), [Communicator](https://us.tobiidynavox.com/products/communicator-5), and [Snap on Windows](https://us.tobiidynavox.com/pages/td-snap). Follow the instructions on the [documentation](https://docs.acecentre.org.uk/products/v/aac-speak-helper-tool) to use this. You will need ['keys'](https://docs.acecentre.org.uk/products/v/aac-speak-helper-tool/getting-keys-for-azure-or-google) (this is a piece of text provided by the provider e.g., Microsoft that allows you to use their software. You generally have to provide credit card details for this step). Our Speak Helper will cache strings of converted audio to allow for offline support if it has spoken that word before but only if it has. (Note: If you are an Ace Centre member of staff please consult Shared Resources for a version with these keys built-in).

    Other engines e.g., [MMS in SherpaOnnx](https://ai.meta.com/blog/multilingual-model-speech-recognition/), allow for offline support. This, too, you can use with our Speak Helper tool. SAPI voices (e.g., [RHVoice](https://rhvoice.org/languages/), [Cereproc](https://www.cereproc.com), [Acapela](https://www.acapela-group.com/demos/), [Nuance](https://www.nuance.com/omni-channel-customer-engagement/voice-and-ivr/text-to-speech/vocalizer.html)) you all will need to [download (and often purchase) Voices](https://nextup.com/ivona/). You will need to visit the respective companies websites or [NextUp](https://nextup.com/) for more information and to hear voices. Note that some software such as [MindExpress](https://www.jabbla.com/en/mindexpress/voices/) have their own pages to download additional voices. We have listed in our list here these specific engines. 
    
    Note what is missing from our list: 
    - iOS-specific voices and apps. Watch this space
    """
    st.markdown(long_text)

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
if engine_filter and engine_filter.lower() != "all":  # Make initial check case-insensitive
    # print("\n=== Engine Filter Debug ===")
    # print(f"Before filtering - total voices: {len(df)}")
    # print(f"Current engine filter: {engine_filter}")
    
    if engine_filter.lower() == "all - except mms":  # Make case-insensitive
        # Debug the filter components separately
        mms_ids = df['id'].str.startswith('mms_', na=False)
        sherpa_engine = df['engine'] == 'sherpaonnx'
        
        # print(f"\nVoices with mms_ IDs: {mms_ids.sum()}")
        # print(f"Voices with sherpaonnx engine: {sherpa_engine.sum()}")
        
        # Apply filter
        filter_mask = ~(mms_ids | sherpa_engine)
        df_filtered = df[filter_mask]
        
        # print(f"\nVoices remaining after filter: {len(df_filtered)}")
        # print("Engines in filtered data:")
        # print(df_filtered['engine'].value_counts())
        
        # Assign filtered data back to df
        df = df_filtered
        
    elif engine_filter.lower() == "all - except espeak":  # Make case-insensitive
        df = df[~df['engine'].str.contains('espeak', case=False)]
    elif engine_filter.lower() == "all - except mms & espeak":  # Make case-insensitive
        df = df[~((df['id'].str.startswith('mms_', na=False)) | 
                 (df['engine'] == 'sherpaonnx') | 
                 df['engine'].str.contains('espeak', case=False))]
    else:
        df = df[df['engine'].str.title() == engine_filter]
    filters_applied = True
if language_search:
    # Convert search term to lowercase
    search_term = language_search.lower()
    # Search in language name, language code, and voice name
    mask = (
        df['language'].str.lower().str.contains(search_term, na=False) | 
        df['language_code'].str.lower().str.contains(search_term, na=False) |
        df['name'].str.lower().str.contains(search_term, na=False)  # Add search in name field
    )
    df = df[mask]
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

    st.markdown(f"There are a **Total {total_voices} Voices Found:**. **Online Voices:** {online_voices} ({online_voices / total_voices:.2%}) **Offline Voices:** {offline_voices} ({offline_voices / total_voices:.2%}) **Male Voices:** {male_voices} ({male_voices / total_voices:.2%}) **Female Voices:** {female_voices} ({female_voices / total_voices:.2%}) **Unknown Gender Voices:** {unknown_gender_voices} ({unknown_gender_voices / total_voices:.2%})")

else:
    st.markdown("**No voices found with the current filter selections.**")

if debug:
    # Debugging: Display first few rows of the dataframe and the engine/status counts
    st.write("Filtered dataframe:", df.head())
    st.write("Engine counts:", df['engine'].value_counts())
    st.write("Status counts:", df['status'].value_counts())

# Display dataframe without certain columns
columns_to_hide = ['latitude', 'longitude', 'name']
df_display = df.drop(columns=columns_to_hide)
st.dataframe(df_display)

# Show map if checkbox is selected
if show_map:
    lang_voices_details = aggregate_voices_by_language(df)
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

# After loading the data
if debug:
    st.write("Sample voice data:", df[['name', 'language', 'language_code']].head(10))

if debug:
    st.write("Sample data:", df[['engine', 'id']].head(10))

# Move the final debug to right after filtering
# if debug and len(df) > 0:
    # print("\n=== Final Data Summary ===")
    # print(f"Total voices in final dataset: {len(df)}")
    # print("\nEngine distribution:")
    # print(df['engine'].value_counts())
    # if len(df) > 0:
    #     print("\nFirst few rows of final dataset:")
    #     print(df[['engine', 'id', 'name']].head())