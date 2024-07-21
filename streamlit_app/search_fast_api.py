from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from pythainlp.corpus import thai_stopwords
import thaispellcheck
import re
from typing import Optional

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to the NHSO Dynamic FAQ API"}

en_stop = set(stopwords.words('english'))
th_stop = set(thai_stopwords())

file_name = 'merged_data.xlsx'
synonyms_file = 'synonymsfile_NHSO.xlsx'

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
        
        expanded_boolean = pd.Series([False] * len(series))
        for expanded_pattern in expanded_patterns:
            expanded_boolean |= series.str.contains(expanded_pattern, regex=False)
        
        boolean_df[pattern] = expanded_boolean
    
    if logic == 'AND':
        result_mask = boolean_df.all(axis=1)
    elif logic == 'OR':
        result_mask = boolean_df.any(axis=1)
    else:
        raise ValueError("Invalid logic: choose 'AND' or 'OR'")
    
    return boolean_df[result_mask]

@app.post("/ask-question")
async def ask_question(question: str = Form(...), logic: str = Form("AND")):
    question = thaispellcheck.check(question, autocorrect=True)
    if os.path.exists(file_name):
        df = pd.read_excel(file_name)
        df.columns = ['Topic', 'Question', 'Answer']
        df = df.iloc[:-1, 1:]
        df['clean_Q'] = df['Question'].apply(clean)
        
        patterns = clean(question).strip().split()
        synonym_dict = load_synonyms(synonyms_file)
        table = search(patterns, df['clean_Q'].to_list(), synonym_dict, logic)
        
        s_index = table.index.to_list()
        if s_index:
            results = []
            for i in s_index:
                results.append({
                    "Question": clean_q(df['Question'][i]),
                    "Answer": df['Answer'][i].strip()
                })
            return {"results": results}
        else:
            return {"message": "ไม่พบข้อมูลที่ตรงกับคำถาม"}
    else:
        return {"message": f"ไม่พบไฟล์ {file_name}"}

@app.post("/view-excel")
async def view_excel(file: str = Form(...)):
    if os.path.exists(file):
        df = pd.read_excel(file)
        return HTMLResponse(df.to_html(), status_code=200)
    else:
        return {"message": f"ไม่พบไฟล์ {file}"}

def update_synonyms(df):
    synonyms_dict = {}
    for _, row in df.iterrows():
        main_word = row['main_word']
        synonyms = set(row['synonyms'].split(', ')) if row['synonyms'] else set()
        synonyms_dict[main_word] = synonyms

    for index, row in df.iterrows():
        current_synonyms = set(row['synonyms'].split(', ')) if row['synonyms'] else set()
        for word in current_synonyms.copy():
            if word in synonyms_dict:
                current_synonyms.update(synonyms_dict[word])
        
        df.at[index, 'synonyms'] = ', '.join(sorted(current_synonyms)) if current_synonyms else ''
    return df

@app.post("/edit-excel")
async def edit_excel(file: UploadFile = File(...)):
    if file.filename not in os.listdir():
        os.makedirs(file.filename, exist_ok=True)
        df_existing = pd.DataFrame(columns=['main_word', 'synonyms'])
    else:
        df_existing = pd.read_excel(file.filename)
    
    uploaded_file = await file.read()
    df_new = pd.read_excel(uploaded_file)
            
    required_columns = ['main_word', 'synonyms']
    if not all(col in df_new.columns for col in required_columns):
        raise HTTPException(status_code=400, detail="ไฟล์ที่อัปโหลดต้องมีคอลัมน์ 'main_word' และ 'synonyms'")
    else:
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined['synonyms'] = df_combined['synonyms'].fillna('')
                
        grouped = df_combined.groupby('main_word').agg({
            'synonyms': lambda x: ', '.join(set(', '.join(x).split(', ')))
        }).reset_index()
                
        grouped = update_synonyms(grouped)
        grouped.to_excel(file.filename, index=False)
        return {"message": "ข้อมูลถูกแก้ไขและบันทึกสำเร็จ"}

@app.post("/save-edited-data")
async def save_edited_data(file_name: str = Form(...), edited_data: UploadFile = File(...)):
    uploaded_file = await edited_data.read()
    with open(file_name, "wb") as f:
        f.write(uploaded_file)
    return {"message": f"บันทึกข้อมูลสำเร็จ: {file_name}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
