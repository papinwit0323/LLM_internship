import streamlit as st
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from pythainlp.corpus import thai_stopwords
import re
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
st.set_page_config(page_title="NHSO Dynamic FAQ", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 5rem;
    }
    .sidebar .sidebar-content .stSelectbox label {
        color: white !important;
    }
    .sidebar .sidebar-content .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: white;
        color: white;
    }
    .sidebar .sidebar-content .stSelectbox div[data-baseweb="select"] > div {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #1E88E5;'>📚 NHSO Dynamic FAQ</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align: center; color: #1E88E5;'>เมนู</h3>", unsafe_allow_html=True)

file_name = 'merged_data.xlsx'

en_stop = set(stopwords.words('english'))
th_stop = set(thai_stopwords())

def clean(text):
    text = text.translate(str.maketrans('', '', '''!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~'''))
    text = text.lower()
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    text = text.replace('\r', '')
    return text

def clean_q(text):
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    text = text.replace('\r', '')
    return text

def load_synonyms(synonyms_file):
    df = pd.read_excel(synonyms_file)
    synonyms = {}
    for index, row in df.iterrows():
        main_term = row['คำหลัก']
        synonym = row['synonyms']
        if main_term not in synonyms:
            synonyms[main_term] = set()
        synonyms[main_term].add(synonym)
        synonyms[main_term].add(main_term)
    return synonyms

def search(patterns, data, synonyms):
    expanded_patterns = set()
    for pattern in patterns:
        if pattern in synonyms:
            expanded_patterns.update(synonyms[pattern])
        expanded_patterns.add(pattern)
    
    series = pd.Series(data)
    counts_df = pd.DataFrame(index=series.index)
    nvocab = len(expanded_patterns)
    
    for pattern in expanded_patterns:
        counts_df[pattern] = series.str.count(pattern)
    
    logit = counts_df > 0
    counts_df = counts_df / logit.sum(axis=0)
    counts_df['tier'] = logit.sum(axis=1)
    counts_df['total_words'] = counts_df.iloc[:, :-1].sum(axis=1)
    counts_df['logit'] = counts_df['tier'] == nvocab
    
    return counts_df.sort_values(by=['tier', 'total_words'], ascending=[False, False])


data = df['Text'].tolist()

# Load synonyms from Excel file
synonyms_file = 'synonyms.xlsx'
synonyms = load_synonyms(synonyms_file)

# Define patterns (primary keywords)
patterns = ["สิทธิบัตรทอง", "สิทธิหลักประกันสุขภาพแห่งชาติ"]

# Perform the search
result = search(patterns, data, synonyms)

# Print the result
print(result)


def upload_and_merge_excel(file_name):
    if os.path.exists(file_name):
        existing_data = pd.read_excel(file_name)
    else:
        existing_data = pd.DataFrame()

    uploaded_file = st.file_uploader("อัปโหลดไฟล์ Excel", type=["xlsx", "xls"])
    if uploaded_file is not None:
        df_new = pd.read_excel(uploaded_file)
        df_combined = pd.concat([existing_data, df_new], ignore_index=True)
        return df_combined
    return existing_data

def view_excel(file_name):
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        st.write(df)
    else:
        st.warning(f"ไม่พบไฟล์ {file_name}")

def edit_excel(file_name):
    df_combined = upload_and_merge_excel(file_name)
    edited_data = st.data_editor(df_combined)
    return edited_data

option = st.sidebar.selectbox(
    'MODE',
    ['🔍 ถามคำถาม', '📊 ดูข้อมูล Excel', '✏️ แก้ไขข้อมูล Excel']
)

if '🔍 ถามคำถาม' in option:
    st.markdown("### 🔍 ถามคำถาม")
    with st.form('my_form'):
        text = st.text_area('ป้อนคำถาม:', 'บิดามีสิทธิเบิกข้าราชการ มีบุตรอายุ 23 ปี แต่ไร้ความสามารถมีความพิการ ต้องการยกเลิกสิทธิข้าราชการเพื่อมาใช้บัตรทองได้หรือไม่')
        submitted = st.form_submit_button('🔎 ค้นหา')
        if submitted:
            with st.spinner('กำลังค้นหาคำตอบ...'):
                if os.path.exists(file_name):
                    df = pd.read_excel(file_name)
                    df.columns = ['Topic', 'Question', 'Answer']
                    df = df.iloc[:-1, 1:]
                    df['clean_Q'] = df['Question'].apply(clean)
                    patterns = clean(text).strip().split()
                    table = search(patterns, df['clean_Q'].to_list())
                    s_index = table[table['logit'] == True].index.to_list()
                    if s_index:
                        for i in s_index:
                            st.write('**Question:**', clean_q(df['Question'][i]))
                            st.write('**Answer:**', df['Answer'][i].strip())
                            st.write('')
                    else:
                        st.write('Not found')
                else:
                    st.warning(f"ไม่พบไฟล์ {file_name}")
                pass

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
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2024 NHSO Dynamic FAQ. All rights reserved.</p>", unsafe_allow_html=True)
