import streamlit as st
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from pythainlp.corpus import thai_stopwords
import thaispellcheck
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
synonyms_file = 'synonymsfile_NHSO.xlsx'
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
    if os.path.exists(synonyms_file):
        df_synonyms = pd.read_excel(synonyms_file)
        synonym_dict = {}
        for index, row in df_synonyms.iterrows():
            main_word = row['main_word']
            synonyms = row['synonyms'].split(',')  # Assuming synonyms are comma-separated
            synonyms = [syn.strip() for syn in synonyms]  # Remove leading/trailing whitespace
            synonym_dict[main_word] = set(synonyms)
            for synonym in synonyms:
                if synonym not in synonym_dict:
                    synonym_dict[synonym] = set()
                synonym_dict[synonym].add(main_word)
                synonym_dict[synonym].update(syn for syn in synonyms if syn != synonym)
        return synonym_dict
    return {}

def expand_pattern_with_synonyms(pattern, synonym_dict):
    expanded_patterns = set()
    if pattern in synonym_dict:
        expanded_patterns.update(synonym_dict[pattern])
    expanded_patterns.add(pattern)
    return expanded_patterns

def search(patterns, data, synonym_dict, logic='AND'):
    series = pd.Series(data)
    boolean_df = pd.DataFrame(index=series.index)
    
    for pattern in patterns:
        expanded_patterns = expand_pattern_with_synonyms(pattern, synonym_dict)
        pattern_boolean = series.str.contains(pattern, regex=False)
        print(expanded_patterns)
        expanded_boolean = pd.Series([False] * len(series))
        for expanded_pattern in expanded_patterns:
            expanded_boolean |= series.str.contains(expanded_pattern, regex=False)
        
        boolean_df[pattern] = expanded_boolean
    
    # Apply the selected logic only to the main patterns
    if logic == 'AND':
        result_mask = boolean_df.all(axis=1)
    elif logic == 'OR':
        result_mask = boolean_df.any(axis=1)
    else:
        raise ValueError("Invalid logic: choose 'AND' or 'OR'")
    return boolean_df[result_mask]





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

def process_input(text, file_name, delimiter=','):
    if file_name not in st.session_state:
        st.error(f"File {file_name} not found in session state.")
        return

    df = st.session_state[file_name]
    words = text.split(delimiter)
    words = [word.strip() for word in words]  # Strip whitespace
    words = list(set(words))  # Remove duplicates

    new_rows = []

    for i, word in enumerate(words):
        main_word = word
        synonyms = [w for j, w in enumerate(words) if i != j]
        synonyms_str = ','.join(synonyms)

        # Check if main_word already exists in the DataFrame
        if main_word in df['main_word'].values:
            idx = df[df['main_word'] == main_word].index[0]
            existing_synonyms = set(df.at[idx, 'synonyms'].split('|'))
            new_synonyms = existing_synonyms.union(set(synonyms))
            df.at[idx, 'synonyms'] = ','.join(sorted(new_synonyms))
        else:
            new_rows.append({'main_word': main_word, 'synonyms': synonyms_str})

    # Append new rows to the DataFrame
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    st.session_state[file_name] = df
    st.experimental_rerun()


def edit_excel1(file_name):
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
                
                grouped = update_synonyms(grouped)

                st.dataframe(grouped)
                if st.button("ใช้ข้อมูลใหม่", key=f"use_new_data_{file_name}"):
                    st.session_state[file_name] = grouped
                    st.experimental_rerun()

    st.subheader("เพิ่มแถวใหม่")
    new_text = st.text_input("ป้อนข้อความใหม่ (คั่นด้วยเครื่องหมาย ','):ตัวอย่างคำค้นหา        >       คำหลัก, synonyms,synonyms เช่น สิทธิบัตรทอง,สิทธิ30บาท,สิทธิหลักประกันสุขภาพถ้วนหน้า")
    if st.button(f"Add New Row to {file_name}"):
        process_input(new_text, file_name)

    if st.button(f"Add Empty Row to {file_name}"):
        new_row = pd.DataFrame([[''] * st.session_state[file_name].shape[1]], columns=st.session_state[file_name].columns)
        st.session_state[file_name] = pd.concat([st.session_state[file_name], new_row], ignore_index=True)
        st.experimental_rerun()

    if not edited_data.empty:
        row_to_delete = st.selectbox(f"Select Row to Delete from {file_name}", edited_data.index)
        if st.button(f"Delete Row from {file_name}"):
            st.session_state[file_name] = edited_data.drop(row_to_delete).reset_index(drop=True)
            st.experimental_rerun()

    return edited_data

def edit_excel(file_name):
    if file_name not in st.session_state:
        st.session_state[file_name] = pd.read_excel(file_name) if os.path.exists(file_name) else pd.DataFrame(columns=['main_word', 'synonyms'])

    st.subheader("อัปโหลดไฟล์ Excel ใหม่")
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ Excel", type=["xlsx", "xls"], key=f"uploader_{file_name}")
    if uploaded_file is not None:
        df_new = pd.read_excel(uploaded_file)

        required_columns = ['ประเด็น','คำถาม', 'คำตอบ']
        if not all(col in df_new.columns for col in required_columns):
            st.error("ไฟล์ที่อัปโหลดต้องมีคอลัมน์ 'ประเด็น','คำถาม', 'คำตอบ'")
        else:
            st.session_state[file_name] = df_new
            

    st.subheader("ตารางปัจจุบัน")
    edited_data = st.data_editor(st.session_state[file_name], width=800)

    # ส่วนการเพิ่มและลบแถว
    st.subheader("ตัวเลือกเพิ่มเติม")
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
def update_search_history(query):
    if query.strip().lower() == 'none':
        return  
    
    history_file = 'search_history1.csv'
    try:
        df = pd.read_csv(history_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['query', 'count'])
    
    if query in df['query'].values:
        df.loc[df['query'] == query, 'count'] += 1
    else:
        new_row = pd.DataFrame({'query': [query], 'count': [1]})
        df = pd.concat([df, new_row], ignore_index=True)
    
    # เรียงลำดับตาม count จากมากไปน้อยและเก็บเฉพาะ 10 อันดับแรก
    df = df.sort_values('count', ascending=False).head(10)
    
    df.to_csv(history_file, index=False)

def load_keywords(file_path):
    try:
        df = pd.read_excel(file_path)

        return df[['ประเด็น', 'คำค้นหา']].set_index('ประเด็น').to_dict()['คำค้นหา']
    except Exception as e:
        st.error(f"ไม่สามารถอ่านไฟล์ keywords ได้: {e}")
        return {}
    
option = st.sidebar.selectbox(
    'MODE',
    ['🔍 ถามคำถาม', '📊 ดูข้อมูล Excel', '✏️ แก้ไขข้อมูล Excel','📊 ดูข้อมูล synonyms_file','✏️ แก้ไขข้อมูล synonyms_file', '📈 ประวัติการค้นหา']
)

if '🔍 ถามคำถาม' in option:
    st.markdown("### 🔍 ถามคำถาม")
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        df.columns = ['Topic', 'Question', 'Answer']
        df = df.iloc[:-1, :]  # ตัดแถวสุดท้ายออก
        
        # โหลด keywords จากไฟล์ Excel อีกไฟล์
        keywords_file = 'key_word.xlsx'  # แทนที่ด้วยพาธของไฟล์ keywords
        keywords_dict = load_keywords(keywords_file)
        
        # สร้าง selectbox สำหรับ Topic
        unique_topics = ['ทั้งหมด'] + sorted(df['Topic'].unique().tolist())
        selected_topic = st.selectbox('เลือกประเด็น (Topic):', unique_topics)
        
        # แสดง keywords ถ้ามี
        if selected_topic != 'ทั้งหมด' and selected_topic in keywords_dict:
            st.markdown(f"**keyword : {keywords_dict[selected_topic]}**")
    else:
        st.warning(f"ไม่พบไฟล์ {file_name}")
        st.stop()  # หยุดการทำงานถ้าไม่พบไฟล์
    
    with st.form('my_form'):
        text = st.text_area('ป้อนคำถาม:', '')
        logic = st.selectbox('เลือกตรรกะการค้นหา', ['AND', 'OR'])
        submitted = st.form_submit_button('🔎 ค้นหา')
    
    if submitted:
        update_search_history(text)
        text = thaispellcheck.check(text, autocorrect=True)
        
        with st.spinner('กำลังค้นหาคำตอบ...'):
            # กรองข้อมูลตาม Topic ที่เลือก
            if selected_topic != 'ทั้งหมด':
                df_filtered = df[df['Topic'] == selected_topic]
            else:
                df_filtered = df
            
            df_filtered['clean_Q'] = df_filtered['Question'].apply(clean)
            
            patterns = clean(text).strip().split()
            synonym_dict = load_synonyms(synonyms_file)
            table = search(patterns, df_filtered['clean_Q'].to_list(), synonym_dict, logic)
            
            s_index = table.index.to_list()
            if s_index:
                for i in s_index:
                    st.write('**Topic:**', df_filtered['Topic'].iloc[i])
                    st.write('**Question:**', clean_q(df_filtered['Question'].iloc[i]))
                    st.write('**Answer:**', df_filtered['Answer'].iloc[i].strip())
                    st.write('')
            else:
                st.write('ไม่พบข้อมูลที่ตรงกับคำถาม')


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
    edited_data = edit_excel1(synonyms_file)
    if st.button('💾 บันทึกข้อมูล'):
        with st.spinner('กำลังบันทึกข้อมูล...'):
            edited_data.to_excel(synonyms_file, index=False)
        st.success(f'บันทึกข้อมูลสำเร็จ: {synonyms_file}')
elif option == '📈 ประวัติการค้นหา':
    st.markdown("### 📈 ประวัติการค้นหา")
    try:
        history_df = pd.read_csv('search_history1.csv')
        if not history_df.empty:
            # เรียงลำดับและแสดงเฉพาะ 10 อันดับแรก
            history_df = history_df.sort_values('count', ascending=False).head(10)
            st.dataframe(history_df)
            
            # สร้างกราฟแท่ง
            import plotly.express as px
            fig = px.bar(history_df, x='query', y='count', title='10 อันดับคำค้นหายอดนิยม')
            fig.update_xaxes(tickangle=45)  # ปรับมุมของข้อความแกน x เพื่อให้อ่านง่ายขึ้น
            st.plotly_chart(fig)
        else:
            st.info("ยังไม่มีประวัติการค้นหา")
    except FileNotFoundError:
        st.info("ยังไม่มีประวัติการค้นหา")
st.markdown("---")
st.markdown("<p style='text-align: center;'>© 2024 NHSO Dynamic FAQ. All rights reserved.</p>", unsafe_allow_html=True)


