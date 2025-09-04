import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from Resume PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text.lower()

# Function to calculate ATS Match Score
def calculate_match_score(resume_text, jd_text):
    cv = CountVectorizer().fit_transform([resume_text, jd_text])
    score = cosine_similarity(cv[0:1], cv[1:2])
    return round(score[0][0] * 100, 2)

# Function to find missing keywords from JD
def find_missing_keywords(resume_text, jd_text):
    jd_keywords = set(jd_text.lower().split())
    resume_words = set(resume_text.lower().split())
    missing = [word for word in jd_keywords if word not in resume_words and len(word) > 2]
    return missing

# --- MAIN PROGRAM ---
# 1. Extract Resume Text
resume_text = extract_text_from_pdf("resume.pdf")

# 2. Load Job Description
with open("job_description.txt", "r", encoding="utf-8") as f:
    jd_text = f.read().lower()

# 3. Calculate Score
score = calculate_match_score(resume_text, jd_text)
print(f"✅ ATS Match Score: {score}%")

# 4. Show Missing Keywords
missing_keywords = find_missing_keywords(resume_text, jd_text)
if missing_keywords:
    print("❌ Missing Keywords:", ", ".join(missing_keywords))
else:
    print("✅ All important keywords are present in the resume!")
