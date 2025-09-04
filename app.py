import streamlit as st
import PyPDF2
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from Resume PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text.lower()

# Function to calculate ATS Match Score
def calculate_match_score(resume_text, jd_text):
    cv = CountVectorizer().fit_transform([resume_text, jd_text])
    score = cosine_similarity(cv[0:1], cv[1:2])
    return round(score[0][0] * 100, 2)

# Function to find missing keywords
def find_missing_keywords(resume_text, jd_text):
    jd_clean = re.sub(r'[^\w\s]', '', jd_text.lower())
    resume_clean = re.sub(r'[^\w\s]', '', resume_text.lower())
    jd_keywords = set(jd_clean.split())
    resume_words = set(resume_clean.split())
    missing = [word for word in jd_keywords if word not in resume_words and len(word) > 2]
    return missing

# Streamlit UI
st.title("📄 ATS Resume Checker")
st.write("Upload your Resume & paste the Job Description to check your ATS Match Score.")

uploaded_file = st.file_uploader("Upload Resume (PDF only)", type="pdf")
jd_text = st.text_area("Paste Job Description Here")

if uploaded_file and jd_text:
    resume_text = extract_text_from_pdf(uploaded_file)
    score = calculate_match_score(resume_text, jd_text)
    st.subheader(f"✅ ATS Match Score: {score}%")

    missing_keywords = find_missing_keywords(resume_text, jd_text)
    if missing_keywords:
        st.error("❌ Missing Keywords: " + ", ".join(missing_keywords))
    else:
        st.success("✅ All important keywords are present in the resume!")
