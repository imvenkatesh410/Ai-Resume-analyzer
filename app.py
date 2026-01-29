import streamlit as st
import spacy
from pdfminer.high_level import extract_text
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# Load NLP model (cached)
@st.cache_resource
def load_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

try:
    nlp = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    nlp = None

def extract_text_from_pdf(pdf_file):
    return extract_text(pdf_file)

def analyze_text(text, nlp_model):
    doc = nlp_model(text)
    # Basic entity extraction (can be improved with custom rules)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def main():
    st.title("📄 AI Resume Analyzer")
    st.markdown("Upload a resume to analyze its content and see suggestions.")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner('Reading Resume...'):
            # Save uploaded file temporarily to ensure compatibility with all versions of pdfminer
            try:
                base_text = extract_text(uploaded_file)
            except Exception as e:
                # Fallback: try saving to temp file if direct stream reading fails
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    base_text = extract_text(tmp_path)
                    os.remove(tmp_path)
                except Exception as inner_e:
                    st.error(f"Error reading PDF: {e} | {inner_e}")
                    base_text = ""

        if base_text:
            st.success("Resume processed successfully!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Raw Text Preview")
                st.text_area("Content", base_text, height=400)

            with col2:
                st.subheader("Analysis Results")
                if nlp:
                    entities = analyze_text(base_text, nlp)
                    
                    # Custom Skill Extraction (Simple keyword matching)
                    # Note: Real AI Resume Analyzers use trained NER models.
                    # Here we use a predefined list for demonstration.
                    common_skills = {
                        "python", "java", "c++", "javascript", "html", "css", "react", "angular", "vue",
                        "sql", "nosql", "aws", "azure", "docker", "kubernetes", "machine learning", "ai",
                        "data analysis", "communication", "leadership", "management", "problem solving"
                    }
                    
                    found_skills = set()
                    doc = nlp(base_text.lower())
                    for token in doc:
                        if token.text in common_skills:
                            found_skills.update([token.text])
                    # Also check for multi-word skills
                    for skill in common_skills:
                        if " " in skill and skill in base_text.lower():
                            found_skills.add(skill)
                            
                    st.write("##### Detected Keywords/Skills")
                    if found_skills:
                        st.write(", ".join([f"`{s.upper()}`" for s in found_skills]))
                    else:
                        st.info("No common skills detected based on simple list.")

                    st.write("##### Named Entities (Spacy)")
                    df = pd.DataFrame(entities, columns=["Entity", "Label"])
                    # Filter helpful entities
                    df_filtered = df[df["Label"].isin(["ORG", "PERSON", "GPE", "DATE"])]
                    st.dataframe(df_filtered, use_container_width=True)

                else:
                    st.warning("NLP Model not loaded.")

            st.subheader("Resume Match Score (Mockup)")
            job_description = st.text_area("Enter Job Description to match against")
            
            if st.button("Calculate Match"):
                if job_description:
                    # Simple keyword overlap (Mock AI)
                    resume_words = set(token.text.lower() for token in nlp(base_text) if not token.is_stop and token.is_alpha)
                    jd_words = set(token.text.lower() for token in nlp(job_description) if not token.is_stop and token.is_alpha)
                    
                    if jd_words:
                        overlap = resume_words.intersection(jd_words)
                        score = len(overlap) / len(jd_words) * 100
                        st.progress(int(score))
                        st.write(f"**Match Score:** {score:.2f}%")
                        st.write("Matching Keywords:", ", ".join(overlap))
                    else:
                        st.warning("Job description is too short.")
                else:
                    st.warning("Please enter a job description.")

if __name__ == "__main__":
    main()
