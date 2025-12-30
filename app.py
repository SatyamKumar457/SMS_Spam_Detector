import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import random

ps = PorterStemmer()

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# SnowFlakes elements

for i in range(300):
    left = random.randint(0, 100)
    size = random.randint(2, 6)
    duration = random.randint(10, 30)
    delay = random.randint(-30, 0)
    opacity = random.random()

    st.html(f"""
    <div class="snowflake" style="
        left:{left}vw;
        width:{size}px;
        height:{size}px;
        animation-duration:{duration}s;
        animation-delay:{delay}s;
        opacity:{opacity};
    "></div>
    """)






st.title("SMS Spam Detection")

input_sms = st.text_area("Check this SMS for spam", key='my_input')

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.download('stopwords')



if st.button('Predict'):

    # 1. Preprocessing
    # 2. Vectorization
    # 3. Prediction
    # 4. Display

    # 1. Preprocessing

    def transform_text(text):
        # Lower case
        text = text.lower()
        # Tokenization
        text = nltk.word_tokenize(text)
        # Removing special characters
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]
        y.clear()
        # Stopwords removal
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
        # Stemming
        text = y[:]
        y.clear()
        for i in text:
            y.append(ps.stem(i))
        return y
        
    transformed_sms = transform_text(input_sms)

    # 2. Vectorization
    # `transform_text` returns a list of tokens; the vectorizer expects raw strings.
    # Join tokens into a single space-separated string before transforming.
    transformed_sms_str = " ".join(transformed_sms)
    vector_input = tfidf.transform([transformed_sms_str])

    # 3. Prediction
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# CSS styling

st.markdown(f"""
<style>
[data-testid="stMain"] {{
    background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
    overflow: hidden;
    display: flex;
    
}}
.snowflake {{
    position: fixed;
    top: -350px;
    background: white;
    border-radius: 50%;
    animation-name: fall;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    
}}
@keyframes fall {{
    to {{
        transform: translate(-700px,110vh);
    }}
}}

[data-testid="stVerticalBlock"]{{
    width: 80%;
    height: auto;
    position : relative;
    margin-top: 150px;
    margin-bottom: 100px;
    padding: 30px 40px 30px 40px;
    border-radius: 15px;
    gap:0px;
    
    
            
    
    background-color: rgba(255, 255, 255, 0.15); 
    backdrop-filter: blur(10px); 
    -webkit-backdrop-filter: blur(10px); 

    border: 1px solid rgba(255, 255, 255, 0.3); 
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1); 
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    animation: glowPulse 5s ease-in-out infinite;
}}
            
@keyframes glowPulse {{
    0%, 100% {{
        box-shadow:
            0 0 25px rgba(80,150,255,0.35),
            0 0 80px rgba(50,120,255,0.25),
            inset 0 1px 1px rgba(255,255,255,0.35);
    }}
    50% {{
        box-shadow:
            0 0 40px rgba(120,180,255,0.45),
            0 0 120px rgba(70,140,255,0.35),
            inset 0 1px 1px rgba(255,255,255,0.45);
    }}
}}
            
[data-testid="stMarkdownContainer"] p {{
    padding: 10px 0 10px 0;
}}
[data-testid="stBaseButton-secondary"]{{
    padding : 0 12px 0 12px;
    margin-top:20px;
}}

            
</style>
            

<div class="snow-container">
    
</div>
""", unsafe_allow_html=True)
