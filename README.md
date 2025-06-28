# Mujid Application

**Mujid** is an innovative Streamlit application that leverages state-of-the-art AI models from **IBM Watson**, **OpenAI**, and **Google Cloud Speech** to process Arabic text and audio. This tool is tailored for Arabic language enthusiasts, providing features like audio recording, text recognition, text-to-speech conversion, and diacritization.

---

## Core Features
- **Multimodal Input**: Supports both voice and text inputs for flexibility and ease of use.
- **Speech-to-Text**: Converts spoken Arabic into written text using Google Cloud's Speech-to-Text API.
- **Arabic Text Diacritization**: Adds diacritical marks to Arabic text for improved clarity and pronunciation using IBM Watson AI.
- **Text-to-Speech**: Converts processed Arabic text into audio using OpenAI models.
- **Enhanced Interface**: Offers a dynamic user experience with customizable backgrounds and component styling.

---

## Installation
Clone the repository and install the required dependencies:
```bash
pip install streamlit ibm-watson openai google-cloud-speech soundfile audio_recorder_streamlit python-dotenv
Setup
Before running the application, ensure the following steps are completed:

API Key Setup:
Obtain API keys from:

IBM Watson

OpenAI

Google Cloud Speech

Environment Variables:
Create a .env file in the project directory and add your keys as follows:

plaintext
Copy
Edit
OPENAI_API_KEY=YOUR_OPENAI_API_KEY  
IBM_WATSONX_API_KEY=YOUR_IBM_API_KEY  
GOOGLE_CLOUD_SPEECH_API_KEY=YOUR_GOOGLE_CLOUD_API_KEY  
USER_ACCESS_TOKEN=YOUR_ACCESS_TOKEN  
How to Run
Navigate to the project directory and run the following command:

bash
Copy
Edit
streamlit run App.py  
Usage
Select your input method: Voice or Text.

For Voice Input: Record your input and submit.

For Text Input: Type your Arabic text and submit.

Outputs include:

Diacritized Text: Arabic text with diacritical marks added.

Audio: Generated speech from the diacritized text.

Key Functions
generate_text: Converts audio input into text using Google Cloud's Speech-to-Text API.

do_Tashkeel: Adds diacritics to Arabic text using IBM Watson's models.

text_to_speech: Transforms text into speech using OpenAI's API.

play_audio: Plays the generated audio file in the Streamlit interface.

Data Preparation
For training and fine-tuning the AI models, we prepared the dataset using a Jupyter Notebook. Key steps include:

Data Loading: Reads Arabic poetry data from a CSV file.

Exploration: Analyzes data distribution and poet diversity.

Libraries Used: pandas, collections, and more for data processing.

Refer to the Data_set_preperation.ipynb file for more details.

Customization
The application allows for aesthetic tweaks, such as setting a custom background image and modifying the component styles through the included utility functions.

Feel free to fork the repository and explore how Mujid combines AI and Arabic language processing to deliver a seamless experience.
