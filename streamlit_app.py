import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import requests


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
        for lang_code in voice['language_codes']:
            if lang_code not in lang_voices_details:
                lang_voices_details[lang_code] = {'count': 0, 'latitudes': [], 'longitudes': []}
            lang_voices_details[lang_code]['count'] += 1
            for lat_long in voice['lat_longs']:
                if lat_long['latitude'] != "0.0" or lat_long['longitude'] != "0.0":  # Check if the location is valid
                    lang_voices_details[lang_code]['latitudes'].append(float(lat_long['latitude']))
                    lang_voices_details[lang_code]['longitudes'].append(float(lat_long['longitude']))
    # Calculate average latitude and longitude for each language
    for lang_code, details in lang_voices_details.items():
        if details['latitudes'] and details['longitudes']:
            avg_lat = sum(details['latitudes']) / len(details['latitudes'])
            avg_long = sum(details['longitudes']) / len(details['longitudes'])
        else:
            avg_lat, avg_long = 0, 0  # Default to 0,0 if no valid locations
        lang_voices_details[lang_code] = {'count': details['count'], 'latitude': avg_lat, 'longitude': avg_long}
    return lang_voices_details

# Streamlit app
st.title("Voice Availability Map")

# Streamlit app adjustments for dataframe creation
voices = get_voices()

# Flatten the latitude and longitude data for each voice
flattened_lat_longs = []
for voice in voices:  # Assuming `voices` is your API response
    for lat_long in voice["lat_longs"]:
        flattened_lat_longs.append({
            "Language": voice["language_codes"][0],  # Example, adjust as needed
            "Voices": 1,  # Assuming each entry counts as one voice
            "Longitude": lat_long["longitude"],
            "Latitude": lat_long["latitude"],
            "Engine": voice["engine"],
            # Add more fields as necessary
        })

# Convert to DataFrame for visualization
df_lat_longs = pd.DataFrame(flattened_lat_longs)

lang_voices_details = aggregate_voices_by_language(voices)
df = pd.DataFrame([(k, v['count'], v['latitude'], v['longitude']) for k, v in lang_voices_details.items()], columns=['Language', 'Voices', 'Latitude', 'Longitude'])

# Interactive Map
st.write("## Voice Availability by Language")
view_state = pdk.ViewState(latitude=0, longitude=0, zoom=1)
max_radius = 20000  # Maximum radius for a circle
scale_factor = 10000  # Adjust this scale factor as needed
# Calculate the radius for each language, ensuring it does not exceed max_radius
df['Radius'] = df['Voices'] * scale_factor
df['Radius'] = df['Radius'].apply(lambda x: min(x, max_radius))

# Update the PyDeck layer to use the pre-calculated radius
layer = pdk.Layer(
    'ScatterplotLayer',
    data=df,
    get_position='[Longitude, Latitude]',
    get_radius='Radius',  # Use the pre-calculated radius column
    get_fill_color='[200, 30, 0, 80]',
    pickable=True
)
tool_tip = {"html": "Language: {Language}<br/>Voices: {Voices}", "style": {"color": "white"}}
map = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tool_tip)
st.pydeck_chart(map)



# Filter Form
st.write("## Filter Voices")
engines_list = ["All", "polly", "google", "microsoft", "watson", "elevenlabs", "witai", "mms", "acapela", "other"]
engine = st.selectbox("Select Engine", options=engines_list)
language = st.selectbox("Select Language", options=['All'] + list(lang_voices_details.keys()), key='language')

# Define selected_language_codes based on the selected language
selected_language_codes = [language] if language != "All" else list(lang_voices_details.keys())

# Adjust filtering for both engine and language
filtered_voices = [voice for voice in voices if (voice["engine"] == engine or engine == "All") and any(lang_code in voice["language_codes"] for lang_code in selected_language_codes)]

# Convert filtered_voices to a DataFrame and display
filtered_voices_df = pd.DataFrame(filtered_voices)
filtered_voices_df = filtered_voices_df.drop(columns=["lat_longs"])  # Remove the lat_longs column
st.write(f"### {len(filtered_voices)} Voices Found")
st.dataframe(filtered_voices_df.style.set_properties(**{'width': '800px', 'height': '400px'}))
