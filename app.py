import streamlit as st
import pdfplumber
import re
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------
# 🧩 Helper Functions
# -----------------------------------------------------

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception:
        text = ""
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf = vectorizer.fit_transform([resume_text, job_desc])
        similarity_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(similarity_score * 100, 2)
    except:
        return 0.0


def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    stopwords = set([
        'with', 'from', 'this', 'that', 'have', 'been', 'will', 'your',
        'and', 'the', 'for', 'you', 'are', 'was', 'but', 'all', 'any'
    ])

    keywords = [w for w in words if w not in stopwords]

    freq = {}
    for word in keywords:
        freq[word] = freq.get(word, 0) + 1

    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    return [word for word, _ in sorted_keywords[:20]]


def generate_pdf_report(score, common_skills, missing_skills):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "ATS Resume Match Report")

    y = height - 1.5 * inch
    c.setFont("Helvetica", 12)

    c.drawString(1 * inch, y, f"ATS Match Score: {score}%")
    y -= 0.4 * inch

    c.drawString(1 * inch, y, "Common Skills (Matched):")
    y -= 0.25 * inch
    for skill in common_skills:
        c.drawString(1.2 * inch, y, f"- {skill}")
        y -= 0.2 * inch

    y -= 0.3 * inch
    c.drawString(1 * inch, y, "Missing Skills (Important for Job):")
    y -= 0.25 * inch
    for skill in missing_skills:
        c.drawString(1.2 * inch, y, f"- {skill}")
        y -= 0.2 * inch

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# -----------------------------------------------------
# 🎯 Streamlit UI
# -----------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screening & Job Match",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Resume Screening & Job Match System")
st.write("Upload your resume and job description to check how well they match! (Now with ATS Report & PDF Download)")


resume_file = st.file_uploader("📄 Upload Resume (PDF or Text)", type=["pdf", "txt"])
job_desc_input = st.text_area("🧾 Paste Job Description", height=200)


# -----------------------------------------------------
# 🚀 Analyze Button
# -----------------------------------------------------
if st.button("🔍 Analyze Match"):

    if resume_file and job_desc_input:

        # Extract Text
        if resume_file.name.endswith(".pdf"):
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = resume_file.read().decode("utf-8")

        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_desc_input)

        # Similarity
        score = get_similarity(resume_clean, job_clean)

        # Keywords
        resume_keywords = extract_keywords(resume_text)
        job_keywords = extract_keywords(job_desc_input)

        common_skills = list(set(resume_keywords) & set(job_keywords))
        missing_skills = list(set(job_keywords) - set(resume_keywords))

        # -----------------------------------------------------------------
        # 📌 Display On-Screen ATS Report
        # -----------------------------------------------------------------
        st.subheader("📊 ATS Match Report")

        st.metric("ATS Resume Match Score", f"{score}%")

        if score > 75:
            st.success("🔥 Strong Resume Match!")
        elif score > 50:
            st.warning("🟡 Moderate Match – Improve alignment.")
        else:
            st.error("🔴 Weak Match – Your resume needs improvement.")

        st.write("---")
        st.subheader("✅ Common Skills (Matched)")
        st.write(", ".join(common_skills) if common_skills else "No matching keywords found.")

        st.subheader("❌ Missing Important Skills (Add These)")
        st.write(", ".join(missing_skills) if missing_skills else "Great! No important skills missing.")

        st.write("---")
        st.subheader("🧠 Resume Keywords")
        st.write(", ".join(resume_keywords))

        st.subheader("🎯 Job Description Keywords")
        st.write(", ".join(job_keywords))

        # -----------------------------------------------------------------
        # 📝 Generate PDF Report
        # -----------------------------------------------------------------

        pdf_buffer = generate_pdf_report(score, common_skills, missing_skills)

        st.download_button(
            label="📥 Download ATS Report (PDF)",
            data=pdf_buffer,
            file_name="ATS_Resume_Report.pdf",
            mime="application/pdf"
        )

    else:
        st.error("❗ Please upload your resume AND paste the job description.")
