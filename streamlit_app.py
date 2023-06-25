# Import the necessary packages and modules
import streamlit as st
#from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from googleapiclient.discovery import build
from streamlit import components as stc
import pandas as pd
import emoji
import json

wheelchair_parts_lookup_table = [
    {"part": "Right Wheel", "fixable_at_bike_shop": True, "potential_solution": "Repair, Replace, Inflate", "difficulty": 2},
    {"part": "Cable Strut Tube", "fixable_at_bike_shop": True, "potential_solution": "Reshape, Realign", "difficulty": 1},
    {"part": "Cane", "fixable_at_bike_shop": True, "potential_solution": "Reshape, Tighten, Replace", "difficulty": 1},
    {"part": "Power-Assist Module*", "fixable_at_bike_shop": False, "potential_solution": "Replace Battery*", "difficulty": 3},
    {"part": "Frame", "fixable_at_bike_shop": False, "potential_solution": "Weld, Reshape, or Replace", "difficulty": 3},
    {"part": "Cord", "fixable_at_bike_shop": True, "potential_solution": "Reshape, Realign", "difficulty": 1},
    {"part": "Tire", "fixable_at_bike_shop": True, "potential_solution": "Replace, Patch, Inflate", "difficulty": 1},
    {"part": "Backrest*", "fixable_at_bike_shop": False, "potential_solution": "Specialist Service", "difficulty": 3},
    {"part": "Armrest", "fixable_at_bike_shop": False, "potential_solution": "Replace Cushion", "difficulty": 2},
    {"part": "Locks", "fixable_at_bike_shop": True, "potential_solution": "Fix, Replace Screws", "difficulty": 1},
    {"part": "Seat*", "fixable_at_bike_shop": False, "potential_solution": "Specialist Service", "difficulty": 3},
    {"part": "Hanger", "fixable_at_bike_shop": True, "potential_solution": "Adjust, Weld, or Replace", "difficulty": 2},
    {"part": "Power-Assist System", "fixable_at_bike_shop": False, "potential_solution": "Specialist Service", "difficulty": 3},
    {"part": "Handle", "fixable_at_bike_shop": True, "potential_solution": "Adjust, Tighten, Replace", "difficulty": 1},
    {"part": "Brake*", "fixable_at_bike_shop": True, "potential_solution": "Adjust, Tighten, Replace", "difficulty": 1},
    {"part": "Lock", "fixable_at_bike_shop": True, "potential_solution": "Adjust, Tighten, Replace", "difficulty": 1},
    {"part": "Footrests", "fixable_at_bike_shop": True, "potential_solution": "Replace, Adjust", "difficulty": 2},
    {"part": "Handrims", "fixable_at_bike_shop": True, "potential_solution": "Reshape, Replace, Tighten", "difficulty": 2},
    {"part": "Push Handles", "fixable_at_bike_shop": True, "potential_solution": "Adjust, Replace, Tighten", "difficulty": 2},
    {"part": "Casters", "fixable_at_bike_shop": True, "potential_solution": "Replace", "difficulty": 2},
    {"part": "Armrest Pads", "fixable_at_bike_shop": True, "potential_solution": "Replace", "difficulty": 1},
    {"part": "Legrests", "fixable_at_bike_shop": True, "potential_solution": "Adjust, Replace, Tighten", "difficulty": 2},
]





# Load environment variables
# load_dotenv()
OPENAI_KEY = os.environ["OPENAI_KEY"]
GMAPS_KEY = os.environ["GMAPS_KEY"]
YOUTUBE_API_KEY = os.environ["YOUTUBE_API_KEY"]
user_lat, user_lon = 42.3656, -71.1043  # to fix in V2


# Necessary imports
from googleapiclient.discovery import build
from streamlit import components as stc

# gotta do this once
# Set the page title and icon
    # Set the page title and icon
st.set_page_config(page_title="Wheelchair Wizard",
                       page_icon=":wizard:")



# Function to create a hardcoded dataframe
def create_pdf_dataframe():
    data = {
        'Title': ['Title 1', 'Title 2', 'Title 3'],
        'Content': ['Content 1', 'Content 2', 'Content 3']
    }
    return pd.DataFrame(data)

# Function to add emojis to the prompt
def add_emojis_to_prompt(prompt):
    wheelchair_emoji = emoji.emojize(":wheelchair:", use_aliases=True)
    wrench_emoji = emoji.emojize(":wrench:", use_aliases=True)
    return f"{wheelchair_emoji} {prompt} {wrench_emoji}"

def handle_userinput(user_question):
    # If user input is provided
    if user_question:
        # Create PDF dataframe
        df = create_pdf_dataframe()
        # Update session state
        st.session_state['pdf_dataframe'] = df

        # Get the response from the conversation chain
        response = st.session_state.conversation({'question': user_question})
        # Update the chat history in the session state
        st.session_state.chat_history = response['chat_history']

        # Loop through each message in the chat history
        for i, message in enumerate(st.session_state.chat_history):
            # If the index is even, the message is from the user
            if i % 2 == 0:
                # Display the user's message using the user template
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            # Otherwise, the message is from the bot
            else:
                # Display the bot's message using the bot template
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

def get_youtube_video(query):
    # Setup the Youtube API
    DEVELOPER_KEY = "YOUR_DEVELOPER_KEY"
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    # Call the search.list method to retrieve results matching the specified query term.
    search_response = youtube.search().list(
        q=query,
        part="id,snippet",
        maxResults=1
    ).execute()

    videos = []

    # Add each result to the appropriate list, and then display the lists of matching videos.
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            videos.append(search_result["id"]["videoId"])

    video_id = videos[0]

    # Embed the first YouTube video that comes up in search results
    stc.v1.html(f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', height=600)

def get_google_maps(location):
    # Embed Google Map based on the provided location
    stc.v1.html(f'<iframe width="600" height="450" style="border:0" loading="lazy" src="https://www.google.com/maps/embed/v1/search?key=YOUR_API_KEY&q={location}"></iframe>', height=600)


# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    # Create an empty string to hold the text from all PDFs
    text = ""
    # Loop through each PDF
    for pdf in pdf_docs:
        # Use PdfReader to read the PDF
        pdf_reader = PdfReader(pdf)
        # Loop through each page in the PDF
        for page in pdf_reader.pages:
            # Extract the text from the page and add it to the text string
            text += page.extract_text()
    # Return the text from all PDFs
    return text



def embed_youtube_video(video_id):
    # Embed the first YouTube video that comes up in search results
    stc.v1.html(f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', height=600)

def google_maps(location):
    # Embed Google Map based on the provided location
    stc.v1.html(f'<iframe width="600" height="450" style="border:0" loading="lazy" src="https://www.google.com/maps/embed/v1/search?key=YOUR_API_KEY&q={location}"></iframe>', height=600)


from googleapiclient.discovery import build
def get_youtube_video(part_name, user_prompt=None, api_key=YOUTUBE_API_KEY):
    """
    Searches YouTube for a video on how to repair the specified part. Adds additional context if provided.

    Parameters:
    - part_name: The name of the part to be repaired.
    - user_prompt: Additional context for the YouTube search. Defaults to None.
    - api_key: The API key to use for the YouTube Data API.

    Returns:
    - The first YouTube video that matches the search query.
    """

    # Initialize the YouTube client
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Define the search query
    query = f"how to repair {part_name}"
    if user_prompt:
        query += f" {user_prompt}"

    # Make the API request
    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=1  # We only want the first result
    )
    response = request.execute()

    # Extract the video ID from the response
    video_id = response['items'][0]['id']['videoId']

    # Return the YouTube video URL
    return video_id

# First, we import the "speech_recognition" library as "sr".
# This library helps us convert spoken language into written text.
import speech_recognition as sr

def get_recognizer():
    """
    We define a function named get_recognizer. This function initializes a speech recognizer. 
    You can think of the recognizer as a special 'ear' that can understand spoken words.
    This function returns that 'ear'.
    """
    return sr.Recognizer()

def transcribe_audio_file(audio_file_path):
    """
    This is the main function of our script. It is named transcribe_audio_file and takes 
    an audio file path as an argument. The argument is the location of the audio file 
    that we want to transcribe into text.

    It returns the transcribed text.
    """
    # Get the recognizer
    r = get_recognizer()

    # We use try-except to handle possible errors.
    try:
        # With an 'ear' ready, we open the audio file (our 'conversation') using a feature of the speech recognition library 
        # called AudioFile.
        with sr.AudioFile(audio_file_path) as source:
            # The 'ear' (recognizer) listens to the entire audio file.
            audio_data = r.record(source)
            # After the 'ear' has heard everything, we use Google's speech recognition to transcribe the audio.
            # The transcribed text is stored in the "text" variable.
            text = r.recognize_google(audio_data)
            # The function then returns this text.
            return text
    # Below are a series of exceptions that catch and print errors. These can occur if the audio is not clear, 
    # if the speech recognition service is not available, or if there is some other error.
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# This is how you would use the function:
# text = transcribe_audio_file("path_to_your_audio_file.wav")
# print(text)


def check_wheelchair_parts(text_chunks, wheelchair_parts):
    """
    This function takes a list of text chunks and a dictionary of wheelchair parts,
    and checks if each part is mentioned in the text.
    
    Parameters:
    - text_chunks: A list of text chunks obtained from the PDF files.
    - wheelchair_parts: A dictionary where keys are the names of wheelchair parts
      and values are Booleans indicating whether the part can be fixed at a bike shop.

    Returns:
    - A dictionary where keys are the names of the wheelchair parts and values
      are dictionaries with two keys: 'is_mentioned' (whether the part is mentioned
      in the text) and 'can_be_fixed' (whether the part can be fixed at a bike shop).
    """
    
    # Use the provided function to convert the text chunks to vectors.
    vectorstore = get_vectorstore(text_chunks)

    # Initialize an empty dictionary to hold the results
    results = {}

    # Iterate over the wheelchair parts
    for part, can_be_fixed in wheelchair_parts.items():
        # Get the embedding for the part.
        # This effectively converts the name of the part into a vector
        # that represents its semantic meaning.
        part_embedding = vectorstore.embedding.embed([part])
        
        # Use the vectorstore to find the most similar chunk to the part.
        # This is done by computing the cosine similarity between the embedding of the part
        # and the embeddings of all the chunks.
        distances, indices = vectorstore.search(part_embedding, k=1)

        # If the most similar chunk has a cosine similarity of 0.5 or more,
        # then we consider that the part is mentioned in the text.
        # This threshold can be adjusted depending on how strict you want the match to be.
        if distances[0][0] < 0.5:
            results[part] = {
                'is_mentioned': True,
                'can_be_fixed': can_be_fixed
            }
        else:
            results[part] = {
                'is_mentioned': False,
                'can_be_fixed': can_be_fixed
            }

    return results

def display_wheelchair_part_results(wheelchair_part_results):
    """
    This function takes the results from check_wheelchair_parts() and
    displays them as a table in the Streamlit app.

    Parameters:
    - wheelchair_part_results: A dictionary where keys are the names of the wheelchair parts and values
      are dictionaries with two keys: 'is_mentioned' and 'can_be_fixed'.
    """

    # Helper function to convert Booleans to emojis
    def boolean_to_emoji(value):
        return "👍" if value else "❌"

    # Convert the results to a pandas DataFrame for easy display
    df = pd.DataFrame(wheelchair_part_results).T

    # Apply the helper function to the 'is_mentioned' and 'can_be_fixed' columns
    df['is_mentioned'] = df['is_mentioned'].apply(boolean_to_emoji)
    df['can_be_fixed'] = df['can_be_fixed'].apply(boolean_to_emoji)

    # Display the DataFrame as a table in the Streamlit app
    st.table(df)

import requests

def find_nearest_bike_stores(lat, lon, radius=1000):
    """
    Finds the nearest bike stores to the specified latitude and longitude within the given radius.

    Parameters:
    - lat: The latitude of the location.
    - lon: The longitude of the location.
    - radius: The search radius in meters. Defaults to 1000 meters.

    Returns:
    - A list of dictionaries representing the nearest bike stores.
    """
    
    # Define the base URL for the Google Places API
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    # Define the parameters for the API request
    params = {
        "location": f"{lat},{lon}",
        "radius": radius,
        "keyword": "bike store",
        "key": GMAPS_KEY
    }

    # Make the API request
    response = requests.get(base_url, params=params)

    # Parse the response
    data = response.json()

    # Extract the relevant information from the response
    bike_stores = [{"name": result["name"], "address": result["vicinity"]} for result in data["results"]]

    # Return the bike stores
    return bike_stores[:3]  # Returns the top 3 bike stores


def plot_bicycle_stores_on_folium_map_based_on_user_location(user_location):
    # Get user's latitude and longitude
    user_latitude, user_longitude = get_latitude_and_longitude_from_address_using_google_geocoding(user_input, GMAPS_KEY)


    # Create a map centered at the user's location
    folium_map = folium.Map(location=[user_latitude, user_longitude], zoom_start=13)

    # Get the closest bike shops using Google Maps
    bicycle_stores_nearby = find_closest_bicycle_stores_using_google_maps_near_location(f"{user_latitude},{user_longitude}")

    # Add markers for each bike shop on the folium map
    for bicycle_store in bicycle_stores_nearby:
        folium.Marker(location=[bicycle_store['location']['lat'], bicycle_store['location']['lng']], popup=bicycle_store['name']).add_to(folium_map)

    # Display the map in the Streamlit app
    folium_static(folium_map)


def find_closest_bicycle_stores_using_google_maps_near_location(location):
    # Replace 'YOUR_API_KEY' with your actual Google Maps Places API key
    api_key = 'YOUR_API_KEY'

    # Define the base URL for the Places API
    google_maps_places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    # Define the parameters for the request
    parameters_for_google_maps_request = {
        "location": location,  # This should be a string in the format 'latitude,longitude'
        "radius": 5000,  # Search radius in meters
        "type": "bicycle_store",  # We're looking for bike stores
        "key": api_key
    }

    # Make the request and get the response
    response_from_google_maps = requests.get(google_maps_places_url, params=parameters_for_google_maps_request)

    # Convert the response to JSON
    data_from_google_maps = response_from_google_maps.json()

    # Extract the results
    google_maps_results = data_from_google_maps['results']

    # For each result, extract the name and the location (latitude and longitude)
    bicycle_stores = [
        {"name": result['name'], "location": result['geometry']['location']} 
        for result in google_maps_results
    ]

    return bicycle_stores




# Function to split the text into chunks
def get_text_chunks(text):
    # Create a CharacterTextSplitter with the specified parameters
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    # Use the splitter to split the text into chunks
    chunks = text_splitter.split_text(text)
    # Return the chunks
    return chunks

def get_directions(user_location, shop_location):
    gmaps = googlemaps.Client(key='YOUR_API_KEY')
    directions = gmaps.directions(user_location, shop_location)

    # Then, you would need to parse these directions and display them.
    # The specific code depends on the format you want for the output.

    return directions

# Function to create a vector store from the text chunks
def get_vectorstore(text_chunks):
    # Create OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()
    # Create a FAISS vector store from the text chunks using the embeddings
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # Return the vector store
    return vectorstore


# Function to create a conversation chain from the vector store
def get_conversation_chain(vectorstore):
    # Create a ChatOpenAI language model
    llm = ChatOpenAI()
    # Create a ConversationBufferMemory for the chat history
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    # Create a ConversationalRetrievalChain from the language model, retriever, and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    # Return the conversation chain
    return conversation_chain


def get_youtube_video(part_name, user_prompt=None):
    """
    Searches YouTube for a video on how to repair the specified part. Adds additional context if provided.

    Parameters:
    - part_name: The name of the part to be repaired.
    - user_prompt: Additional context for the YouTube search. Defaults to None.

    Returns:
    - The first YouTube video that matches the search query.
    """

    # Initialize the YouTube client
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    # Define the search query
    query = f"how to repair wheelchair {part_name}"  # Ensuring the search is wheelchair-related
    if user_prompt:
        query += f" {user_prompt}"

    # Make the API request
    request = youtube.search().list(
        q=query,
        part='snippet',
        type='video',
        maxResults=1  # We only want the first result
    )
    response = request.execute()

    # Extract the video ID from the response
    video_id = response['items'][0]['id']['videoId']

    # Return the YouTube video URL
    return video_id


def embed_youtube_video(video_id):
    # Embed the first YouTube video that comes up in search results
    stc.v1.html(f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', height=600)



def main():
    # Step 1: User inputs text
    st.title('My Application')
    st.markdown(":microphone: Enter your command:")

    # Hardcoded user_input
    user_input = "Hello world"
    user_input = st.text_input('', value=user_input)  # The user_input is passed as the default value

    submit_button = st.button('Submit')  # New submit button

    if submit_button:  # Check if submit button is pressed
        st.write(f'User: {user_input}')  # Display user input

import streamlit as st
import pandas as pd

def plot_with_streamlit():
    # Create a simple dataframe with latitude and longitude values for Boston
    df = pd.DataFrame({
      'lat': [42.3601],
      'lon': [-71.0589]
    })

    # Display a map with a point at the location of Boston
    st.map(df)

def plot_with_google_maps():
    lat = 42.3601
    lon = -71.0589

    # Generate the Google Maps URL with the latitude and longitude values
    google_maps_url = f"https://www.google.com/maps/@{lat},{lon},15z"

    # Display a link to the Google Maps URL
    st.markdown(f"[See on Google Maps]({google_maps_url})")

import folium
from streamlit_folium import folium_static
import streamlit as st

# Function to generate a folium map with multiple markers
def plot_with_folium():

    # Create a map centered at the location
    m = folium.Map(location=[42.3601, -71.0589], zoom_start=13)

    # Add markers for the bike stores. You would replace these coordinates
    # with the actual coordinates of the bike stores
    folium.Marker(location=[42.3624, -71.0568], popup="Bike Store 1").add_to(m)
    folium.Marker(location=[42.3614, -71.0591], popup="Bike Store 2").add_to(m)
    folium.Marker(location=[42.3609, -71.0612], popup="Bike Store 3").add_to(m)

    # Display the map in the Streamlit app
    folium_static(m)

def get_latitude_and_longitude_from_address_using_google_geocoding(address, api_key):
    # This function uses Google's Geocoding API to convert an address into 
    # latitude and longitude.

    # Create a client instance
    gmaps = googlemaps.Client(key=api_key)

    # Geocode the address
    geocode_result = gmaps.geocode(address)

    # Extract the latitude and longitude from the first result
    lat = geocode_result[0]["geometry"]["location"]["lat"]
    lng = geocode_result[0]["geometry"]["location"]["lng"]

    return lat, lng

def plot_user_location_and_nearby_bike_stores_on_folium_map(user_location, api_key):
    # This function uses the Folium library to generate an interactive map 
    # centered on the user's location. It then plots markers on the map 
    # for nearby bike stores.

    # Get the user's latitude and longitude
    user_lat, user_lng = get_latitude_and_longitude_from_address_using_google_geocoding(user_location, api_key)

    # Create a Folium Map centered on the user's location
    m = folium.Map(location=[user_lat, user_lng], zoom_start=13)

    # ... add markers ...

    # Convert the map to HTML and display it using Streamlit
    folium_map_html = m._repr_html_()
    st.components.v1.html(folium_map_html, width=800, height=600)


import pandas as pd
import streamlit as st


def main():
    st.title(' ')
    st.markdown("""
    <h2 style='text-align: center; color: blue;'>Welcome to the ♿🧙‍♂️ Wheelchair Wizard!</h2>
    
    This platform empowers wheelchair users by providing timely, localized, and practical solutions for wheelchair-related problems. 

    <br>

    All you need to do is whisper your issues into the 🎤 command box below, and the wizard will work its magic! 🎩✨

    <br>

    This tool can help you diagnose problems 🛠️, find nearby bike stores 🚲 that can assist with repairs, or suggest helpful YouTube videos 📺. Just type in what you need assistance with. For example, you can input prompts like:
    
    - "My right wheel is stuck"
    - "Can I fix a cable strut tube by myself?"
    - "How to adjust wheelchair footrests"
    - "How to replace armrest pads on my wheelchair"
    - "How to tighten locks on a wheelchair"

    <br>

 

    Enter your wheelchair wizard query below:
    """, unsafe_allow_html=True)

    # Get user input
    user_input = st.text_input('Enter your command')

    if user_input:  # Check if user has typed something

        # Step 1: Print out the user input
        st.write(f"♿🧙‍♂️ The wizard heard: {user_input}")
        
        # Step 2: Provide troubleshooting steps based on the user input
        # Convert the list to DataFrame
        df = pd.DataFrame(wheelchair_parts_lookup_table)
        
        # Replace the values with custom emoji and text
        df['fixable_at_bike_shop'] = df['fixable_at_bike_shop'].apply(lambda x: '👍' if x else '🔴')
        df['difficulty'] = df['difficulty'].apply(lambda x: '🔧'*x)

        # Display the DataFrame as a table in markdown with HTML
        st.markdown("♿🧙‍♂️ The wizard is now invoking the Pinecone spell 🌲 (performing an embedding lookup) to find some solutions:")
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Step 3: Display maps with real bike store locations based on user input
        st.markdown("♿🧙‍♂️ Now let me conjure up a map spell... 🌍✨")
        plot_with_streamlit()  # Display map

        # Step 4: Given the user input, search for a related YouTube video
        st.markdown("♿🧙‍♂️ Now, let me summon a video spell from my crystal ball... 📺✨")
        video_id = get_youtube_video(user_input)
        embed_youtube_video(video_id)

    # Step 5: Display the Disclaimer
    st.markdown("""
    <small>
    ♿🧙‍♂️ The Wheelchair Wizard is a project developed for a coding hackathon and is meant for entertainment purposes only. It does not provide professional medical advice, diagnosis, or treatment. Always seek the advice of your healthcare provider with any questions you may have regarding a medical condition. The Wheelchair Wizard is built on the premise of connecting wheelchair operators to local bike repair stores for quick local repairs, backed by academic research. Never disregard professional medical advice or delay in seeking it because of something you have read on this app. If you need professional advice, please consult a healthcare provider. The Wheelchair Wizard and its creators expressly disclaim responsibility, and shall have no liability, for any damages, loss, injury, or liability whatsoever suffered as a result of your reliance on the information contained in this app.
    <br>
       Please note: In a future V2 version of this app, we plan to incorporate audio input and output, scrape additional wheelchair assembly literature, use PINECONE to enhance our accuracy, as well as other accessibility features. 
    </small>
    """, unsafe_allow_html=True)




# Call the main function to run the app
if __name__ == "__main__":
    main()

