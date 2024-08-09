from elasticsearch import Elasticsearch
from transformers import RagTokenizer, RagSequenceForGeneration
import streamlit as st
import PyPDF2

# Set up the Streamlit interface
st.title("AI-Powered Knowledge Management System")
st.write("Upload PDF documents containing practitioner knowledge and ask tailored questions.")

# Initialize Elasticsearch
es = Elasticsearch()

# Process PDFs
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Upload PDF
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    documents = [extract_text_from_pdf(file) for file in uploaded_files]
    st.write("Extracted text from the uploaded PDFs.")
    
    # Index the documents in Elasticsearch
    for i, doc in enumerate(documents):
        es.index(index="documents", id=i, document={"content": doc})
    
    # Input question from the researcher
    question = st.text_input("Enter your question:")

    # Load the RAG model
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

    # Function to process question and retrieve answer
    def answer_question(question):
        # Search Elasticsearch for relevant documents
        results = es.search(index="documents", query={"match": {"content": question}})
        contexts = [hit["_source"]["content"] for hit in results["hits"]["hits"]]
        
        inputs = tokenizer(question, contexts, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    if st.button("Get Answer"):
        if question:
            response = answer_question(question)
            st.write("Answer:", response)
        else:
            st.write("Please enter a question.")
