import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Carica le variabili d'ambiente
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("La variabile OPENAI_API_KEY non è stata caricata correttamente.")
    st.stop()

# Imposta il percorso del vector DB persistente
persist_directory = "vectordb"
if not os.path.exists(persist_directory):
    st.error("Il vector DB non è stato trovato. Esegui prima l'ingestione con 'ingest.py'.")
    st.stop()

# Carica il vector DB abilitando la deserializzazione per file pickle
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)

# Inizializza il modello LLM e la catena di RetrievalQA
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Assistente per Mutui e Finanziamenti", layout="wide")
st.title("Assistente per Mutui e Finanziamenti")

st.subheader("Fai una domanda su mutui, finanziamenti, ecc.")
user_input = st.text_input("Inserisci la tua domanda qui")

if st.button("Invia") and user_input:
    with st.spinner("Generazione della risposta..."):
        answer = qa_chain.invoke(user_input)
        # Se il risultato è un dizionario, estrai solamente il valore associato a "result"
        if isinstance(answer, dict) and "result" in answer:
            answer = answer["result"]
    st.markdown(f"**Q:** {user_input}")
    st.markdown(f"**A:** {answer}")
