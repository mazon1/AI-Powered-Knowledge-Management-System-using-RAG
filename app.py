import streamlit as st
import PyPDF2
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# Set up the Streamlit interface
st.title("AI-Powered Knowledge Management System")
st.write("Upload PDF documents containing practitioner knowledge and ask tailored questions.")

# Upload PDF
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Process PDFs
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfFileReader(file)
    text = ""
    for page in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page).extract_text()
    return text

if uploaded_files:
    documents = [extract_text_from_pdf(file) for file in uploaded_files]
    st.write("Extracted text from the uploaded PDFs.")

# Input question from the researcher
question = st.text_input("Enter your question:")

# Load the RAG model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

# Function to process question and retrieve answer
def answer_question(question, documents):
    # Create context from documents
    contexts = documents
    inputs = tokenizer(question, contexts, return_tensors="pt", padding=True, truncation=True)
    
    # Generate response
    outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if st.button("Get Answer"):
    if question and documents:
        response = answer_question(question, documents)
        st.write("Answer:", response)
    else:
        st.write("Please upload documents and enter a question.")
