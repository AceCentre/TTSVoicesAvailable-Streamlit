import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import requests
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import csv
import plotly.express as px
import plotly.graph_objects as go

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

# Add this function to load the language data
def load_language_data():
    try:
        # Load the CSV file
        language_data = pd.read_csv('alltts.csv')
        # Clean up and prepare the data
        language_data = language_data.rename(columns={
            'ISO 693-3': 'iso_code',
            'Language Name': 'language_name',
            'Population Text': 'population_text',
            'Population Estimate': 'population',
            'ethnologuelink': 'ethnologue_link'
        })
        
        # Extract the actual URL from the HTML link
        def extract_url(html_link):
            if pd.isna(html_link):
                return ""
            import re
            url_match = re.search(r'href="([^"]+)"', html_link)
            if url_match:
                return url_match.group(1)
            return ""
        
        # Extract the link text
        def extract_link_text(html_link):
            if pd.isna(html_link):
                return ""
            import re
            text_match = re.search(r'>([^<]+)<', html_link)
            if text_match:
                return text_match.group(1)
            return "Ethnologue"
        
        # Create clean URL and text columns
        language_data['ethnologue_url'] = language_data['ethnologue_link'].apply(extract_url)
        language_data['ethnologue_text'] = language_data['ethnologue_link'].apply(extract_link_text)
        
        # Convert population to numeric, handling non-numeric values
        language_data['population'] = pd.to_numeric(language_data['population'], errors='coerce')
        return language_data
    except Exception as e:
        st.error(f"Error loading language data: {str(e)}")
        # Print more details for debugging
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# Add this after the dataframe display
st.header("Language Coverage Analysis")
show_language_coverage = st.checkbox("Show Language Coverage Chart", value=False)

if show_language_coverage:
    # Add filter checkboxes
    col1, col2 = st.columns(2)
    with col1:
        exclude_mms = st.checkbox("Exclude MMS voices", value=False)
    with col2:
        exclude_espeak = st.checkbox("Exclude eSpeak voices", value=False)
    
    # Load language data
    language_data = load_language_data()
    
    if not language_data.empty:
        # Create a progress indicator
        progress_bar = st.progress(0)
        st.info("Analyzing language coverage... This may take a moment.")
        
        # Create a filtered dataframe based on checkbox selections
        filtered_df = df.copy()
        
        # Debug information
        st.write(f"Total voices before filtering: {len(filtered_df)}")
        
        # Apply filters if selected
        if exclude_mms:
            filtered_df = filtered_df[~((filtered_df['id'].str.startswith('mms_', na=False)) | 
                                      (filtered_df['engine'] == 'sherpaonnx'))]
            st.write(f"Total voices after excluding MMS: {len(filtered_df)}")
        
        if exclude_espeak:
            filtered_df = filtered_df[~filtered_df['engine'].str.contains('espeak', case=False)]
            st.write(f"Total voices after excluding eSpeak: {len(filtered_df)}")
        
        # Get unique languages from voice data
        voice_languages = filtered_df['language'].dropna().unique()
        voice_language_codes = filtered_df['language_code'].dropna().unique()
        
        st.write(f"Number of unique languages in voice data: {len(voice_languages)}")
        st.write(f"Number of unique language codes in voice data: {len(voice_language_codes)}")
        
        # Count voices by language
        language_voice_counts = filtered_df.groupby('language').agg({
            'id': 'count',
            'status': lambda x: (x == 'Online').sum()
        }).rename(columns={
            'id': 'total_voices',
            'status': 'online_voices'
        })
        
        # Count MMS and eSpeak voices by language
        mms_counts = df[(df['id'].str.startswith('mms_', na=False)) | 
                        (df['engine'] == 'sherpaonnx')].groupby('language').size()
        espeak_counts = df[df['engine'].str.contains('espeak', case=False)].groupby('language').size()
        
        language_voice_counts['mms_voices'] = mms_counts.reindex(language_voice_counts.index, fill_value=0)
        language_voice_counts['espeak_voices'] = espeak_counts.reindex(language_voice_counts.index, fill_value=0)
        
        # Create a summary dataframe for visualization
        summary_data = pd.DataFrame({
            'Category': ['Languages in the World', 'Languages with TTS', 
                        'Languages with Online TTS', 'Languages with Offline TTS',
                        'Languages with Quality TTS (non-MMS, non-eSpeak)'],
            'Count': [
                len(language_data),
                len(language_voice_counts),
                (language_voice_counts['online_voices'] > 0).sum(),
                # Count languages with offline voices - fixed the error here
                filtered_df[filtered_df['status'] == 'Offline']['language'].nunique(),
                # Count languages with quality voices (non-MMS, non-eSpeak)
                ((language_voice_counts['total_voices'] > 0) & 
                 (language_voice_counts['mms_voices'] == 0) & 
                 (language_voice_counts['espeak_voices'] == 0)).sum()
            ]
        })
        
        # Calculate percentages
        summary_data['Percentage'] = summary_data['Count'] / summary_data['Count'][0]
        
        # Display summary metrics
        st.subheader("TTS Language Coverage Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Languages in the World", summary_data['Count'][0])
        col2.metric("Languages with TTS", 
                   summary_data['Count'][1], 
                   f"{summary_data['Percentage'][1]:.1%}")
        col3.metric("Languages with Online TTS", 
                   summary_data['Count'][2], 
                   f"{summary_data['Percentage'][2]:.1%}")
        col4.metric("Languages with Offline TTS", 
                   summary_data['Count'][3], 
                   f"{summary_data['Percentage'][3]:.1%}")
        col5.metric("Languages with Quality TTS", 
                   summary_data['Count'][4], 
                   f"{summary_data['Percentage'][4]:.1%}")
        
        # Create a bar chart for the summary
        fig = px.bar(
            summary_data[1:],  # Skip the first row (total languages)
            x='Category',
            y='Count',
            color='Category',
            color_discrete_map={
                'Languages with TTS': 'orange',
                'Languages with Online TTS': 'lightgreen',
                'Languages with Offline TTS': 'yellow',
                'Languages with Quality TTS': 'darkgreen'
            },
            title='TTS Language Coverage',
            text='Count'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
        
        # Create a list of languages without TTS
        languages_with_tts = set(language_voice_counts.index)
        languages_without_tts = language_data[~language_data['language_name'].isin(languages_with_tts)]
        
        # Show top languages without TTS
        st.subheader("Languages Without TTS")
        st.write(f"Top 50 languages by population without TTS support (out of {len(languages_without_tts)}):")
        st.dataframe(
            languages_without_tts.sort_values('population', ascending=False)[
                ['language_name', 'iso_code', 'population', 'ethnologue_url']
            ].head(50),
            column_config={
                "ethnologue_url": st.column_config.LinkColumn("Ethnologue Link"),
                "population": st.column_config.NumberColumn(format="%d")
            }
        )
        
        # Complete progress
        progress_bar.progress(100)
    else:
        st.error("Could not load language data. Please check if the CSV file exists.")