import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Detection")

input_sms = st.text_area("Enter the message", key='my_input')


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')



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