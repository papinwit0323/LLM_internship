import streamlit as st
from PIL import Image

# ฟังก์ชั่นสำหรับตอบกลับของ Chatbot
def chatbot_response(user_input):
    # คุณสามารถเพิ่มฟังก์ชั่นสำหรับการประมวลผลคำตอบของ chatbot ที่นี่
    return f"คุณพิมพ์ว่า: {user_input}"

# ตั้งค่าหน้าเว็บของ Streamlit
st.set_page_config(page_title="Chatbotสั่งควยไรนักหนาวะเพ้อเจ้อไอเหี้ย", page_icon="🤖", layout="wide")

# แสดงโลโก้หรือรูปภาพที่หัวเว็บ (ถ้ามี)
st.image("C:\LLM_internship\streamlit_app\image_app\คาสึยะ.jpg", width=400)  # ใช้ URL หรือไฟล์โลโก้ของคุณเอง


# ตั้งค่าสไตล์
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 10px;
    }
    .stTextInput, .stButton, .stTextArea {
        font-size: 18px;
        padding: 10px;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #333333;
        border: 2px solid #cccccc;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stTextArea > div > textarea {
        background-color: #ffffff;
        color: #333333;
        border: 2px solid #cccccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# แสดงหัวเรื่องของเว็บ
st.title("Chatbotสั่งควยไรนักหนาวะเพ้อเจ้อไอเหี้ย")

# แสดงข้อความแนะนำการใช้งาน
st.write("กรุณาพิมพ์ข้อความของคุณที่นี่ และกด 'ส่ง' เพื่อรับคำตอบจาก Chatbot")

# สร้างช่องรับข้อความจากผู้ใช้
user_input = st.text_input("กรุณาพิมพ์ข้อความของคุณที่นี่")

# ถ้าผู้ใช้กดปุ่ม 'ส่ง' ให้แสดงผลลัพธ์
if st.button('ส่ง'):
    response = chatbot_response(user_input)
    st.text_area("Chatbot ตอบ:", value=response, height=200)

# เพิ่ม footer หรือข้อความเพิ่มเติมที่ส่วนล่างของหน้าเว็บ
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px;">
        <hr>
        <p>Powered by Streamlit | Developed by papinwit,nattakit,pirawit</p>
    </div>
    """,
    unsafe_allow_html=True
)
