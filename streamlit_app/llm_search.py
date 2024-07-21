import streamlit as st
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from pythainlp.corpus import thai_stopwords
import re
import numpy as np
from pythainlp import word_tokenize
from pythainlp.util import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
st.set_page_config(page_title="NHSO Dynamic FAQ", page_icon="🌟", layout="wide")

st.markdown("""
<style>
    body {
        background-color: #f0f4f8;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .main {
        padding: 3rem;
        background-color: white;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border: 2px solid #bdc3c7;
        border-radius: 15px;
        padding: 0.8rem;
        transition: all 0.3s ease;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        padding: 2rem;
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 3rem;
    }
    .sidebar h3 {
        color: white;
        font-size: 1.8rem;
        margin-bottom: 2rem;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .stSelectbox label {
        color: white !important;
        font-weight: 600;
    }
    .sidebar .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.2);
        border: none;
        border-radius: 10px;
        color: white;
    }
    .sidebar .stSelectbox div[data-baseweb="select"] > div {
        color: white;
    }
    div.row-widget.stRadio > div {
        flex-direction: row;
        align-items: stretch;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label {
        background-color: #f0f2f6;
        padding: 10px 20px;
        margin: 5px;
        border-radius: 5px;
        flex-grow: 1;
        text-align: center;
        transition: background-color 0.3s, box-shadow 0.3s;
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label:hover {
        background-color: #e0e2e6;
        box-shadow: 0 0 5px rgba(0,0,0,0.2);
    }
    div.row-widget.stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    div.row-widget.stRadio > div[role="radiogroup"] input:checked + label {
        background-color: #2e7bcf;
        color: white;
        font-weight: bold;
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌟 NHSO LLM WITH RAG</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3>Menu</h3>", unsafe_allow_html=True)
file_name = 'merged_data.xlsx'
synonyms_file = 'synonymsfile_NHSO.xlsx'


os.environ["OPENTYPHOON_API_KEY"] = 'sk-RBNtvMyKIyTk9G5J1J3OegmdfD03y6v3Pp9sPvcdPuCEOsr8'
@st.cache_data
def load_data():
    df = pd.read_excel('merged_data.xlsx')
    return df

df = load_data()

def preprocess_thai_text(text):
    text = normalize(text)
    tokens = word_tokenize(text, keep_whitespace=False)
    return ' '.join(tokens)

df['processed_question'] = df['คำถาม'].apply(preprocess_thai_text)


vectorizer = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
tfidf_matrix = vectorizer.fit_transform(df['processed_question'])

def find_similar_questions(query, top_k=3):
    processed_query = preprocess_thai_text(query)
    query_vec = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return df.iloc[top_indices]

def generate_response(input_text):
    if os.environ["OPENTYPHOON_API_KEY"].startswith('sk-'):
        client = OpenAI(api_key=os.environ["OPENTYPHOON_API_KEY"], base_url="https://api.opentyphoon.ai/v1")
        
        #ค้นหาคำถามในไฟล์นั่นแหละ
        similar_questions = find_similar_questions(input_text)
        
        # สร้างcontextไว้ใช้ข้างล่าง
        context = "\n\n".join([f"ประเด็น: {row['ประเด็น']}\nคำถาม: {row['คำถาม']}\nคำตอบ: {row['คำตอบ']}" for _, row in similar_questions.iterrows()])
        
        # สร้างพร้อมเกรียนสำหรับส่งไปยัง API
        prompt = f"""โปรดใช้ข้อมูลต่อไปนี้เป็นบริบทในการตอบคำถาม:

{context}

คำถามของผู้ใช้: {input_text}

กรุณาตอบคำถามโดยใช้ข้อมูลจากบริบทข้างต้น หากไม่มีข้อมูลที่เกี่ยวข้องโดยตรง โปรดใช้ความรู้ทั่วไปในการตอบ 
โปรดตอบเป็นภาษาไทยที่เป็นทางการและสุภาพ"""

        response = client.chat.completions.create(
            model="typhoon-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.7,
            top_p=1,
        )
        st.info(response.choices[0].message.content)
    else:
        st.warning('กรุณาใส่ OpenAI API Key ของคุณ!', icon='⚠')
def view_excel(file_name):
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        st.write(df)
    else:
        st.warning(f"ไม่พบไฟล์ {file_name}")

def update_synonyms(df):
    # สร้างดิกชันนารีของ synonyms สำหรับทุก main_word
    synonyms_dict = {}
    for _, row in df.iterrows():
        main_word = row['main_word']
        synonyms = set(row['synonyms'].split(', ')) if row['synonyms'] else set()
        synonyms_dict[main_word] = synonyms

    # อัปเดต synonyms สำหรับทุกแถว
    for index, row in df.iterrows():
        current_synonyms = set(row['synonyms'].split(', ')) if row['synonyms'] else set()
        for word in current_synonyms.copy():  # ใช้ .copy() เพื่อหลีกเลี่ยงการเปลี่ยนแปลงขณะวนลูป
            if word in synonyms_dict:
                current_synonyms.update(synonyms_dict[word])
        
        # อัปเดตคอลัมน์ synonyms
        df.at[index, 'synonyms'] = ', '.join(sorted(current_synonyms)) if current_synonyms else ''

    return df

def edit_excel(file_name):
    if file_name not in st.session_state:
        st.session_state[file_name] = pd.read_excel(file_name) if os.path.exists(file_name) else pd.DataFrame(columns=['main_word', 'synonyms'])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ตารางปัจจุบัน")
        edited_data = st.data_editor(st.session_state[file_name])

    with col2:
        st.subheader("อัปโหลดและรวมข้อมูลใหม่")
        uploaded_file = st.file_uploader("อัปโหลดไฟล์ Excel", type=["xlsx", "xls"], key=f"uploader_{file_name}")
        if uploaded_file is not None:
            df_new = pd.read_excel(uploaded_file)
            
            required_columns = ['main_word', 'synonyms']
            if not all(col in df_new.columns for col in required_columns):
                st.error("ไฟล์ที่อัปโหลดต้องมีคอลัมน์ 'main_word' และ 'synonyms'")
            else:
                df_combined = pd.concat([st.session_state[file_name], df_new], ignore_index=True)
                df_combined['synonyms'] = df_combined['synonyms'].fillna('')
                
                grouped = df_combined.groupby('main_word').agg({
                    'synonyms': lambda x: ', '.join(set(', '.join(x).split(', ')))
                }).reset_index()
                
                # อัปเดต synonyms
                grouped = update_synonyms(grouped)

                st.dataframe(grouped)
                if st.button("ใช้ข้อมูลใหม่", key=f"use_new_data_{file_name}"):
                    st.session_state[file_name] = grouped
                    st.experimental_rerun()

    # ส่วนการเพิ่มและลบแถว
    if st.button(f"Add Row to {file_name}"):
        new_row = pd.DataFrame([[''] * st.session_state[file_name].shape[1]], columns=st.session_state[file_name].columns)
        st.session_state[file_name] = pd.concat([st.session_state[file_name], new_row], ignore_index=True)
        st.experimental_rerun()

    if not edited_data.empty:
        row_to_delete = st.selectbox(f"Select Row to Delete from {file_name}", edited_data.index)
        if st.button(f"Delete Row from {file_name}"):
            st.session_state[file_name] = edited_data.drop(row_to_delete).reset_index(drop=True)
            st.experimental_rerun()

    return edited_data
option = st.sidebar.radio(
    'MODE',
    ['🔍 ถามคำถาม', '📊 ดูข้อมูล Excel', '✏️ แก้ไขข้อมูล Excel', '📊 ดูข้อมูล synonyms_file', '✏️ แก้ไขข้อมูล synonyms_file'],
    format_func=lambda x: x.split()[-1]  # แสดงเฉพาะข้อความหลังอิโมจิ
)

# แสดงผลลัพธ์
st.write(f"คุณเลือก: {option}")

if '🔍 ถามคำถาม' in option:
    with st.form('my_form'):
        text = st.text_area('ป้อนคำถาม:', '3 เคล็ดลับสำคัญในการเรียนรู้การเขียนโปรแกรมคืออะไร?')
        submitted = st.form_submit_button('ยืนยัน')
        if submitted:
            generate_response(text)



elif option == '📊 ดูข้อมูล Excel':
    st.markdown("### 📊 ดูข้อมูล Excel")
    view_excel(file_name)
elif option == '✏️ แก้ไขข้อมูล Excel':
    st.markdown("### ✏️ แก้ไขข้อมูล Excel")
    edited_data = edit_excel(file_name)    
    if st.button('💾 บันทึกข้อมูล'):
        with st.spinner('กำลังบันทึกข้อมูล...'):
            edited_data.to_excel(file_name, index=False)
        st.success(f'บันทึกข้อมูลสำเร็จ: {file_name}')
elif option == '📊 ดูข้อมูล synonyms_file':
    st.markdown("### 📊 ดูข้อมูล synonyms_file")
    view_excel(synonyms_file)
elif option == '✏️ แก้ไขข้อมูล synonyms_file':
    st.markdown("### ✏️ แก้ไขข้อมูล synonyms_file")
    edited_data = edit_excel(synonyms_file)
    if st.button('💾 บันทึกข้อมูล'):
        with st.spinner('กำลังบันทึกข้อมูล...'):
            edited_data.to_excel(synonyms_file, index=False)
        st.success(f'บันทึกข้อมูลสำเร็จ: {synonyms_file}')
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2024 NHSO Dynamic FAQ. All rights reserved.</p>", unsafe_allow_html=True)