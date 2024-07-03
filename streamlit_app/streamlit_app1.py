import streamlit as st
from langchain_community.llms import OpenAI
import numpy as np
from rank_bm25 import BM25Okapi
import re
import os
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize as thai_word_tokenize
from collections import Counter

# Setup the Streamlit app
st.title('🦜🔗 Quickstart App')

# Sidebar for API Key input
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

# Download NLTK resources
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize stopwords
en_stop = set(stopwords.words('english'))
th_stop = set(thai_stopwords())

# Initialize stemmer and lemmatizer
p_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

# Define the text cleaning function
def clean(text):
    text = text.translate(str.maketrans('', '', '''!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~'''))
    text = text.lower()
    text = text.replace('\n', '')
    text = ' '.join(text.split())
    return text

# Define the iterative word splitting function
def split_word_iterative(tokens):
    result = []
    for token in tokens:
        # Clean the token
        token = clean(token)

        # Remove stop words in Thai and English
        if token not in th_stop and token not in en_stop:
            # Stem English words
            token = p_stemmer.stem(token)

            # Find the Thai word roots using WordNet
            w_syn = wordnet.synsets(token)
            if len(w_syn) > 1 and len(w_syn[0].lemma_names('tha')) > 1:
                token = w_syn[0].lemma_names('tha')[0]

            # Remove numbers
            if not token.isnumeric():
                # Remove spaces
                if ' ' not in token and any(token) and '"' not in token:
                    result.append(token)
    return result

# Define the word splitting function
def split_word(text):
    # Tokenize the text
    tokens = thai_word_tokenize(text, engine='newmm')

    # Iteratively process tokens
    return split_word_iterative(tokens)

# Sample data for demonstration
df_document = {
    'document': [
        "ซีซันที่ 1 ออกอากาศเมื่อ 9 เมษายน 2567 ที่ญี่ปุ่นทาง โดยสร้างโดย TOHO studio",
        "ทาง Muse Thailand นำเข้าอนิเมะซีซันที่ 1 ออกอากาศหลังทางญี่ปุ่น 7 วัน ที่ไทย",
        "หมาบินไม่ได้นะ เด็กๆ",
        "ส่วนทางซีซันที่ 2 ของไคจูหมายเลข8 มีแผนจะฉายในปีหน้า ที่ญี่ปุ่นที่แรก",
        "the ghost radio เป็นช่องที่เล่าเรื่องผีที่ยอดนิยม"
    ],
    'doc_id': [1, 2, 3, 4, 5]
}
df_question = {
    'question': ["ไคจูหมายเลข8ซีซั่นแรกออกอากาศเมื่อวันที่เท่าไหร่?"],
    'doc_id': [1]
}

# Tokenize documents and questions
tokenized_context = [split_word(doc) for doc in df_document['document']]
doc_context_id = df_document['doc_id']

question_id = df_question['doc_id']
tokenized_question = [split_word(q) for q in df_question['question']]

# Define the BM25 ranking function
def rank_b25(token_query, token_doc, docs):
    bm25 = BM25Okapi(token_doc)
    sim_score = bm25.get_scores(token_query)
    index = np.argsort(sim_score)[::-1]
    return zip(index[:5], [re.sub('\r?\n', ' ', docs[i]) for i in index[:5]])

# Generate relevant text from BM25
relevant = rank_b25(tokenized_question[0], tokenized_context, df_document['document'])
relevant_text = '{ '
for idx, i in relevant:
    relevant_text += f'{idx}: "{i}", '
relevant_text = relevant_text[:-2]
relevant_text += ' }'

# Example for prompt and system prompt
ex_doc = '{1: "ซีซันที่ 1 ออกอากาศเมื่อ 9 เมษายน 2567 ที่ญี่ปุ่นทาง โดยสร้างโดย TOHO studio", 2:"ทาง Muse Thailand นำเข้าอนิเมะซีซันที่ 1 ออกอากาศหลังทางญี่ปุ่น 7 วัน ที่ไทย", 3:"หมาบินไม่ได้นะ เด็กๆ", 4: "ส่วนทางซีซันที่ 2 ของไคจูหมายเลข8 มีแผนจะฉายในปีหน้า ที่ญี่ปุ่นที่แรก", 5:"the ghost radio เป็นช่องที่เล่าเรื่องผีที่ยอดนิยม"}'
ex_q = 'ไคจูหมายเลข8ซีซั่นแรกออกอากาศเมื่อวันที่เท่าไหร่?'
ex_aws = 'ซีซันที่ 1 ออกอากาศเมื่อ 9 เมษายน 2567 ที่ญี่ปุ่น และออกอากาศในวันที่ 14 เมษายน 2567 ทางประเทศไทย [1],[2]'

sys_prompt = f"You are a helpful assistant that must try your best effort to answer the user question ALWAYS following this guidelines:\nKeep your answer grounded in the facts provided in REAL DOCUMENT section and insert REAL DOCUMENTS ID that support your REAL ANSWER.\nIf the DOCUMENT section doesn’t contain the facts to answer the QUESTION ALWAYS return [NONE].\nFor example:\nEXAMPLE DOCUMENT: {ex_doc}\nEXAMPLE QUESTION: {ex_q}\nEXAMPLE ANSWER: {ex_aws}"
query_prompt = 'REAL DOCUMENT: {}\nREAL QUESTION: {}\nREAL ANSWER:'.format(relevant_text, df_question['question'][0])
print(sys_prompt)
print(query_prompt)

# Function to generate response using OpenAI
def generate_response(input_text):
    os.environ["OPENTYPHOON_API_KEY"] = openai_api_key
    client = OpenAI(api_key=os.environ["OPENTYPHOON_API_KEY"], base_url="https://api.opentyphoon.ai/v1")
    st.info(client(input_text))

# Streamlit form to get user input and submit
with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)
