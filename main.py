import streamlit as st
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Configuração do modelo para embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Configuração do modelo LLM
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Função para processar um PDF
def process_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Função para dividir texto em chunks
def split_text_into_chunks(text, chunk_size=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Configuração da interface Streamlit
st.title("Assistente Conversacional AS05")

uploaded_files = st.file_uploader("Envie arquivos PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.info("Processando os arquivos...")
    all_chunks = []
    all_texts = []

    for file in uploaded_files:
        # Extrai texto do PDF
        extracted_text = process_pdf(file)
        all_texts.append(extracted_text)

        # Divide o texto em chunks
        chunks = split_text_into_chunks(extracted_text)
        all_chunks.extend(chunks)

    st.write(f"Total de chunks processados: {len(all_chunks)}")

    # Gera embeddings para todos os chunks
    embeddings = model.encode(all_chunks)
    st.success("Embeddings gerados com sucesso!")

    # Armazenamento no FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    st.success("Embeddings indexados com sucesso!")

    # Implementação de busca
    question = st.text_input("Faça uma pergunta:")
    if question:
        # Gera embedding da pergunta
        question_embedding = model.encode([question])
        
        # Busca no índice
        distances, indices = index.search(question_embedding, k=5)  # Retorna os 5 mais próximos
        relevant_chunks = [all_chunks[i] for i in indices[0]]

        # Exibe os chunks relevantes
        st.write("Chunks mais relevantes:")
        for chunk in relevant_chunks:
            st.write(chunk)

        # Usar LLM para resposta final
        context = " ".join(relevant_chunks)
        answer = qa_pipeline(question=question, context=context)
        st.write("Resposta gerada pelo modelo:")
        st.write(answer['answer'])
