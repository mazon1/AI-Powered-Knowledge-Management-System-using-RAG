import streamlit as st
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up the Streamlit interface
st.title("AI-Powered Knowledge Management System")
st.write("Upload PDF documents containing practitioner knowledge and ask tailored questions.")

# Process PDFs
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Upload PDF
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = [extract_text_from_pdf(file) for file in uploaded_files]
    st.write("Extracted text from the uploaded PDFs.")
    
    # Input question from the researcher
    question = st.text_input("Enter your question:")

    # Load models
    @st.cache_resource
    def load_models():
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        return tokenizer, model, sentence_model

    tokenizer, model, sentence_model = load_models()

    # Function to process question and retrieve answer
    def answer_question(question, documents):
        # Encode the question and documents
        question_embedding = sentence_model.encode([question])[0]
        document_embeddings = sentence_model.encode(documents)

        # Find the most similar document
        similarities = cosine_similarity([question_embedding], document_embeddings)[0]
        most_similar_idx = np.argmax(similarities)
        context = documents[most_similar_idx]

        # Generate answer
        inputs = tokenizer([question + " " + context], max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=30, max_length=200)
        return tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                response = answer_question(question, documents)
                st.write("Answer:", response)
        else:
            st.write("Please enter a question.")
else:
    st.write("Please upload PDF files to proceed.")
