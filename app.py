import streamlit as st
import os
import io
import base64
from ibm_watsonx_ai.foundation_models import Model
from google.cloud import speech
import soundfile as sf
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from openai import OpenAI
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials

load_dotenv()

def get_watsonx_response(model, prompt):
    response = model.generate_text(prompt=prompt)
    return response

def generate_text(audio_bytes, api_key):
    temp_file_path = r"C:\Users\moham\Desktop\GUILAST\temp_audio.wav"
    
    try:
        # Load the audio from bytes and convert it to mono
        with io.BytesIO(audio_bytes) as audio_file:
            data, samplerate = sf.read(audio_file)
            if data.ndim > 1:  # Check if it's stereo
                data = data.mean(axis=1)  # Convert to mono by averaging the channels

            # Save the converted mono audio as a .wav file
            sf.write(temp_file_path, data, samplerate)

        # Initialize Google Cloud Speech client with API key
        client = speech.SpeechClient(client_options={"api_key": api_key})
        
        with io.open(temp_file_path, "rb") as audio_file:
            content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="ar-JO"  # Arabic (Jordan)
        )

        response = client.recognize(request={"config": config, "audio": audio})

        # Extracting and returning the recognized text
        for result in response.results:
            return result.alternatives[0].transcript

    except Exception as e:
        # Handle file creation or recognition errors
        print(f"Error processing audio: {e}")
        return None

    finally:
        # Clean up the temporary file if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
def initialize_Allam_BaseModel():
    credentials = Credentials(
        url="https://eu-de.ml.cloud.ibm.com",
        api_key=os.getenv("IBM_WATSONX_API_KEY")
    )
    params = {
        "decoding_method": "greedy",
        "max_new_tokens": 500,
        "temperature":0.0,
        "stop_sequences": ["\n\n"],    
        "repetition_penalty": 1.2,
        "random_seed":42
    } 
    model = Model(
        model_id='sdaia/allam-1-13b-instruct',
        params=params,
        credentials=credentials,
        project_id="901a08bf-56e9-406b-9d52-a28e57468744"
    )
    return model
def do_Tashkeel(text):
    template = initialize_prompt_for_tashkeel()
    full_prompt = template.format(text=text)
    try:
        llm = initialize_Allam_BaseModel()
        output = get_watsonx_response(llm, full_prompt)
        return output
    except Exception as e:
        st.error(f"Error during processing: {e}")
        return None
    
def initialize_prompt_for_tashkeel():
    template = """
    **مهمتك:** تشكيل النصوص العربية المكتوبة بشكل دقيق وصحيح لضمان الوضوح اللغوي وضمان النطق السليم.

    **التوجيهات:**
    
    - بناءًا على المدخل قم بتوليد المخرج المناسب له
    - لا تقم بتغيير النصوص الأصلية للشعر المقدم ولا تقم بأضافة أي تعليقات أو نصوص توضيحية إضافية.
    - أضف تشكيل فقط لتحسين الوضوح والنطق.
    - فقط قم في الإجابة بالمخرج ولا شيء زيادة
    - لا تقوم بأعادة التوجيهات

    **تعليمات صارمة:**
     إضافة أي تعليقات أو نصوص توضيحية إضافية.

    **أمثلة على المدخل والمخرج المقابل**

    المثال الأول
    input:
    سلي يا عبل عمرا عن فعالي
    بأعداك الأولى طلبوا قتالي
    سليهم كيف كان لهم جوابي
    إذا ما فال ظنك في مقالي
    أتونا في الظلام على جياد
    مضمرة الخواصر كالسعالي
    وفيهم كل جبار عنيد
    شديد البأس مفتول السبال

    output:
    سَلي يا عَبلَ عَمراً عَن فِعالي
    بِأَعداكِ الأُولى طَلَبوا قِتالي
    سَليهُم كَيفَ كانَ لَهُم جَوابي
    إِذا ما فالَ ظَنُّكِ في مَقالي
    أَتَونا في الظَلامِ عَلى جِيادٍ
    مُضَمَّرَةِ الخَواصِرِ كَالسَعالي
    وَفيهِم كُلُّ جَبّارٍ عَنيدٍ
    شَديدِ البَأسِ مَفتولِ السِبالِ
    
    المثال الثاني:
    input: 
    لهذا اليوم بعد غد اريج
    ونار في العدو لها اجيج
    تبيت بها الحواصن آمنات
    وتسلم في مسالكها الحجيج
    فلا زالت عداتك حيث كانت
    فرائس ايها الاسد المهيج
    عرفتك والصفوف معبآت
    وانت بغير سيفك لا تعيج
    ووجه البحر يعرف من بعيد
    اذا يسجو فكيف اذا يموج

    output: 
    لِهَذا اليَومِ بَعدَ غَدٍ أَريجٍ
    وَنارٌ في العَدوِّ لَها أَجيجُ
    تَبيتُ بِها الحَواصِنُ آمِناتٍ
    وَتَسلَمُ في مَسالِكِها الحَجيجُ
    فَلا زالَت عُداتُكَ حَيثُ كانَت
    فَرائِسَ أَيُّها الأَسَدُ المَهيجُ
    عَرَفتُكَ وَالصُفوفُ مُعَبَّآتٌ
    وَأَنتَ بِغَيرِ سَيفِكَ لا تَعيجُ
    وَوَجهُ البَحرِ يُعرَفُ مِن بَعيدٍ
    إِذا يَسجو فَكَيفَ إِذا يَموجُ

    
    المثال الثالث:
    input: 
    بارض تهلك الاشواط فيها
    اذا ملئت من الركض الفروج
    تحاول نفس ملك الروم فيها
    فتفديه رعيته العلوج
    ابالغمَرات توعدنا النصارى
    ونحن نجومها وهي البروج
    وفينا السيف حملته صدوق
    اذا لاقى وغاَرته لجوج

    output: 
    بِأَرضٍ تَهلِكُ الأَشواطُ فيها
    إِذا مُلِئَت مِنَ الرَكضِ الفُروجُ
    تُحاوِلُ نَفسَ مَلكِ الرومِ فيها
    فَتَفديهِ رَعِيَّتُهُ العُلوجُ
    أَبِالغَمَراتِ توعِدُنا النَصارى
    وَنَحنُ نُجومُها وَهِيَ البُروجُ
    وَفينا السَيفُ حَملَتُهُ صَدوقٌ
    إِذا لاقى وَغارَتُهُ لَجوجُ

    input: {text}

    output:
    
    """
    return template

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def text_to_speech(text):
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="shimmer",
            input=text
        )
        audio_bytes = response.content
        if audio_bytes is None:
            raise ValueError("No audio content returned from API")
        return audio_bytes
    except Exception as e:
        st.error(f"Failed to generate speech: {e}")
        return None

def play_audio(audio_bytes):
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")

def initialize_Mujid():
    access_token = os.environ['USER_ACCESS_TOKEN']

    generate_params = {
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.MIN_NEW_TOKENS:100,
    GenParams.DECODING_METHOD:"greedy",
    GenParams.REPETITION_PENALTY:1.2}

    wml_credentials = {
                   "url": "https://ai.deem.sa/",
                   "token": access_token,
                   "instance_id": "openshift",
                   "version": "5.0"
                  }

    model_inference = ModelInference(
            deployment_id="dd7cbd1c-084c-4d2b-8322-435505fc7973",
            params=generate_params,
            credentials=wml_credentials,
            project_id="a6a54a4e-659e-4b98-9068-95bcdb331709"
            )
    
    return model_inference


def generateResultsFromFineTuned(llm, userQues):

    template= """
        مهمتك: كتابة القصائد العربية بأوزان الشعر الكلاسيكي وقوافيه المحددة مسبقاً، مع الحفاظ على الجودة اللغوية والبلاغية للشعر العربي.

        التوجيهات:

        1. استخدم البحر الشعري المحدد بدقة في جميع الأبيات، مع مراعاة تناسق الأوزان والتناغم الصوتي بين الأبيات.
        2. طبق القافية المطلوبة بشكل دقيق في نهاية كل بيت، مع الحفاظ على التسلسل المنطقي للأصوات والمعاني.
        3. استخدم لغة فصيحة تتميز بالرقي والجزالة، مع الالتزام بأسلوب الشعر العربي الكلاسيكي الذي يتسم بالتصوير البلاغي والمحسنات اللفظية.
        4. حافظ على تماسك الموضوع وانسجام الأفكار داخل النص، مع التأكيد على العاطفة أو الفلسفة المطلوبة.
        5. ضمِّن النص عناصر تعكس روح الشعر العربي التقليدي، وأضف مراجع ثقافية وتاريخية تعزز النص وتعمق الصلة بالموروث الأدبي.
        6. تأكد من أن القصيدة تتكون من خمسة أبيات بالضبط، مما يتطلب مهارة في التكثيف والدقة في توصيل المعنى والعاطفة.

        أمثلة على المدخل والمخرج المقابل:

        المثال الأول:

        المدخل:
          اكتب قصائد مدح على بحر المتقارب وقافية ف.
        
        المخرج: 
        صرفنا الأعنة نحو المدام *** وما للزمان علينا صروف
        فغابت كواكب لذاتنا *** وأعجل شمس المدام الكسوف
        فجد بالتي عندها للسرور *** حياة وللهم فيها حتوف
        فما جاد بالراح إلا الجواد *** وما كبر الظرف إلا الظريف
    
        المثال الثاني:

        المدخل:
          اكتب قصائد فراق على بحر المنسرح وقافية د.
        
        المخرج: 
        أ أقحوانا أرته أم بردا *** غيداء يهتز عطفها غيدا
        رنت إليه بطرف خاذلة *** ضعيفة الطرف تضعف الجلدا
        لو وجدت للفراق ما وجدا *** لافتقدت نومها كما افتقدا
        لا تلح صبا على صبابته *** وإن رأى الغي في الهوى رشدا
        فلم تزل للفراق غائلة *** تلذ في المورد الذي وردا
        
         المدخل:
         {UserQuestion}

         المخرج:

        """
    full_prompt = template.format(UserQuestion=userQues)
    Response=llm.generate(full_prompt)
    generated_text = Response['results'][0]['generated_text'].strip()
    return generated_text


def initalize_promptForAllam():
    template = """
        دورك الأساسي هو **"إعادة صياغة سؤال المستخدم"** فقط وفقاً للتعليمات التالية، ولا يجب عليك تحت أي ظرف من الظروف توليد أي شعر أو نص آخر.
        
        ملاحظة بالغة الأهمية:
        - **ممنوع منعاً باتاً توليد أي شعر.**
        - مهمتك الوحيدة هي إعادة صياغة السؤال وفقًا للتعليمات. إذا خالفت هذا الشرط، فأنت لا تؤدي وظيفتك.
     
        الجزء الأول: 
        إذا طلب المستخدم قصيدة دون تحديد البحر أو القافية، فأعد صياغة السؤال بالطريقة التالية: 
        - اكتب قصيدة على بحر (اختيار بحر من القائمة) وقافية (اختيار قافية من القائمة). 

        إليك بعض الأمثلة:
        Input: اعطيني شعر عن الحب
        Output: اكتب قصائد غزل على بحر البسيط وقافية ن
        Input: ابغى شعر حزين
        Output: اكتب قصائد حزينة على بحر الكامل وقافية م
        Input: اريد شعر قصير
        Output: اكتب قصائد قصيرة على بحر الطويل وقافية د

        ### البحور الشعرية المتاحة: 
        1. الطويل
        2. المديد
        3. البسيط
        4. الوافر
        5. الكامل
        6. الرجز
        7. المتقارب
        8. السريع
        9. المنسرح
        10. الخفيف

        ### القوافي المتاحة:
        1. م
        2. ن
        3. ل
        4. ب
        5. ر
        6. د
        7. ت
        8. س
        9. ف
        10. ك
       
        الجزء الثاني:
        - في حال قام المستخدم بتزويدك بالبحر او القافيه او كلاهما ف لا تقم بتغير أي شيئ
        
          الآن دورك
            المستخدم: {user_question}
        """ 
    return template

def rephrasing(user_question):
    template = initalize_promptForAllam()
    full_prompt = template.format(user_question=user_question)
    try:
        llm = initialize_Allam_BaseModel()
        response = get_watsonx_response(llm, full_prompt)
        return response
    except Exception as e:
        st.error(f"Error during processing: {e}")
        return None

import base64

# Set page configuration to wide mode
st.set_page_config(layout="wide")

def get_base64_of_file(file_path):
    """Utility function to get base64 string of a file."""
    with open(file_path, "rb") as file:
        data = base64.b64encode(file.read()).decode('utf-8')
    return data

def set_bg_image(image_file):
    """Utility function to set a full background image."""
    bg_image_base64 = get_base64_of_file(image_file)
    bg_image_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_image_base64}");
            background-size: cover;  /* Changes from 'contain' to 'cover' to ensure it fills the entire background */
            background-repeat: no-repeat;
            background-position: center center;
            background-attachment: fixed;  /* Keeps the background image fixed during scrolling */
            min-height: 100vh;  /* Ensures the background covers the viewport height */
        }}
        </style>
        """
    st.markdown(bg_image_style, unsafe_allow_html=True)

def set_custom_styles():
    """Utility function to set custom styles for Streamlit components."""
    custom_styles = f"""
    <style>
    /* Set all text elements to white */
    body, .stTextInput, .stTextArea, .stMarkdown, h1, h2, h3, h4, h5, h6, label, button, .caption, .css-1s6u09g, .css-18e3th9 {{
        color: white !important;
    }}
    /* Adjust text input and text area components for visibility */
    .stTextInput > div > div > input, .stTextArea > div > textarea {{
        color: black !important;
        background-color: #333 !important; /* Adjust as needed for visibility */
    }}
    /* Modify placeholder colors to be lighter */
    ::placeholder {{ /* Chrome, Firefox, Opera, Safari 10.1+ */
        color: #bbb !important;
    }}
    :-ms-input-placeholder {{ /* Internet Explorer 10-11 */
        color: #bbb !important;
    }}
    ::-ms-input-placeholder {{ /* Microsoft Edge */
        color: #bbb !important;
    }}
    </style>
    """
    st.markdown(custom_styles, unsafe_allow_html=True)

# Path to your image
background_image = r"C:\Users\moham\Desktop\GUILAST\GUI\back3.png"

# Applying the background image
set_bg_image(background_image)

# Apply custom styles after setting the background image
set_custom_styles()

# Your Streamlit application logic follows here
st.title("مُجيد (Mujid)")



# Initialize user_question to None
user_question = None

# Streamlit interface
input_method = st.radio("Choose input method:", ("Voice", "Text"))
if input_method == "Voice":
    st.write("Please record your question:")
    audio_bytes = audio_recorder()
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        user_question = generate_text(audio_bytes, os.getenv('GOOGLE_CLOUD_SPEECH_API_KEY'))
else:
    user_question = st.text_area("Enter your poetry question:", height=150)

if user_question:
    st.write("Processing your question...")
    refreezedInput = rephrasing(user_question)

    if refreezedInput:
        st.markdown("**Rephrasing Question:**")
        st.text_area("Rephrasing Question",value=refreezedInput, height=150, key="output")
        Mujiad = initialize_Mujid()
        st.markdown("generating....")
        Mujiad_response = generateResultsFromFineTuned(Mujiad, refreezedInput)
        print("Mujiad_response\n",Mujiad_response)
        st.text_area("Mujiad Response",value=Mujiad_response, height=150, key="output1")
        st.write("adding Tashkeel....")
        generatedPoetwith_Tashkeel= do_Tashkeel(Mujiad_response)
        st.text_area("generated Poem with Tashkeel",value=generatedPoetwith_Tashkeel, height=150, key="output2")        # Example usage in your Streamlit app
        
        if generatedPoetwith_Tashkeel:
            st.write("Converting text to speech...")
            audio_bytes = text_to_speech(generatedPoetwith_Tashkeel)
            play_audio(audio_bytes)
