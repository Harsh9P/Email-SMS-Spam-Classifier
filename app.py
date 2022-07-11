import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


st.subheader('')
st.subheader('')
st.subheader('')
st.markdown('**@ Created By Harsh Patel**.')
st.subheader('')
st.subheader('')
st.markdown('**Some Examples for Spam messages. To get Idea to write Input text. Or you can Copy & Paste it to check how model will Run.**')
st.subheader('')
st.markdown('**Spam Messages**')
st.markdown('`>> Congratulations you won 1000 dollars. Call on this number to get your prize.`')
st.markdown('`>> You could be entitled up to $100K in compensation from Canada Tax Department as tax credit. Please reply CTC for info or STOP to opt out.`')
st.markdown('**Not Spam Messages**')
st.markdown('`>> I am free today, Lets go out for a movie. What do you say?`')
st.markdown('`>> Did you see the match? It was insane.`')