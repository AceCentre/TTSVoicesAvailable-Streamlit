# TTSVoicesAvailable-Streamlit
 Streamlit App to display details on all TTS available. Uses our API to fetch details on them all 

See at https://ttsvoicesavailable.streamlit.app

Be warned - until we fix some of the issues from tts-wrapper this wont work great.. 

## Data Collection Scripts

Our API needs to have geographical data on the TTS voices available. We have a script that can be run to collect this data. 

```bash
cd data_collection_scripts
python get_langcodes.py
```

We might need to clean the data up. if So look at update_langs.py and gpt_gen_langs.py. NB: You'll need an azure open ai key for the second part. We really need to clean this whole geocoding step up. 

We then move that to the API repo. (Its far more efficient to do this in one go rather than do this in streamlit)