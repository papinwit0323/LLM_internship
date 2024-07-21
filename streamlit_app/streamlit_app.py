import streamlit as st
from openai import OpenAI
import os
import pandas as pd

st.title('NHSO Dynamic FAQ')
#openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password'
#os.environ["OPENTYPHOON_API_KEY"] = openai_api_key
os.environ["OPENTYPHOON_API_KEY"] = 'sk-RBNtvMyKIyTk9G5J1J3OegmdfD03y6v3Pp9sPvcdPuCEOsr8'
def generate_response(input_text):
    if os.environ["OPENTYPHOON_API_KEY"].startswith('sk-'):
        client = OpenAI(api_key=os.environ["OPENTYPHOON_API_KEY"], base_url="https://api.opentyphoon.ai/v1")
        response = client.chat.completions.create(
            model="typhoon-instruct",
            messages=[{"role": "user", "content": input_text}],  # User input question
            max_tokens=1500,
            temperature=0.7,
            top_p=1,
        )
        st.info(response.choices[0].message.content)
    else:
        st.warning('กรุณาใส่ OpenAI API Key ของคุณ!', icon='⚠')

# Function to upload and merge Excel file
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

# Function to edit Excel data
def edit_excel(file_name):
    df_combined = upload_and_merge_excel(file_name)
    edited_data = st.data_editor(df_combined)
    return edited_data

# Navigation bar
option = st.sidebar.selectbox(
    'เลือกโหมด',
    ['ถามคำถาม', 'ดูข้อมูล Excel', 'แก้ไขข้อมูล Excel']
)




if option == 'ถามคำถาม':
    with st.form('my_form'):
        text = st.text_area('ป้อนคำถาม:', '3 เคล็ดลับสำคัญในการเรียนรู้การเขียนโปรแกรมคืออะไร?')
        submitted = st.form_submit_button('ยืนยัน')
        if submitted:
            generate_response(text)
elif option == 'ดูข้อมูล Excel':
    st.subheader('ข้อมูลที่มีอยู่ใน merged_data.xlsx')
    view_excel('merged_data.xlsx')
elif option == 'แก้ไขข้อมูล Excel':
    st.subheader('แก้ไขข้อมูลใน merged_data.xlsx')
    edited_data = edit_excel('merged_data.xlsx')
    
    if st.button('บันทึกข้อมูล'):
        edited_data.to_excel('merged_data.xlsx', index=False)
        st.success('บันทึกข้อมูลสำเร็จ: merged_data.xlsx')
