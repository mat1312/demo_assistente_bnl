import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit.components.v1 as components

# Carica le variabili d'ambiente dal file .env
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

# Inizializza il modello LLM e la catena di RetrievalQA includendo il ritorno dei documenti sorgente
llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # oppure un altro chain_type a seconda delle tue esigenze
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Assistente per Mutui e Finanziamenti", layout="wide")
st.title("Assistente per Mutui e Finanziamenti")

# ==============================
# SEZIONE 1: Q&A tramite LangChain e OpenAI
# ==============================
st.subheader("Fai una domanda su mutui, finanziamenti, ecc.")
user_input = st.text_input("Inserisci la tua domanda qui")

if st.button("Invia") and user_input:
    with st.spinner("Generazione della risposta..."):
        result = qa_chain.invoke(user_input)
        if isinstance(result, dict) and "result" in result:
            answer = result["result"]
            source_docs = result.get("source_documents", [])
    st.markdown(f"**Q:** {user_input}")
    st.markdown(f"**A:** {answer}")

    # Visualizza le fonti come link cliccabili se disponibili, mostrando anche il nome del file, la pagina e la riga
    if source_docs:
        st.markdown("**Fonti:**")
        # Aggrega per fonte per evitare duplicati, accumulando le occorrenze (pagina e riga)
        sources_dict = {}
        for doc in source_docs:
            metadata = doc.metadata
            if "source" in metadata:
                # Normalizza il percorso sostituendo backslash con slash
                source = metadata["source"].replace("\\", "/")
                page = metadata.get("page", None)
                line = metadata.get("start_index", None)
                if source in sources_dict:
                    sources_dict[source].append((page, line))
                else:
                    sources_dict[source] = [(page, line)]
                    
        # Visualizza ogni fonte come link cliccabile con il nome del file e le occorrenze (pagina e riga)
        for source, occurrences in sources_dict.items():
            file_name = os.path.basename(source)
            occ_list = []
            for occ in occurrences:
                p, l = occ
                occ_str = ""
                if p is not None:
                    occ_str += f"pagina {p}"
                if l is not None:
                    occ_str += f", riga {l}" if occ_str else f"riga {l}"
                if occ_str:
                    occ_list.append(occ_str)
            occ_text = " - ".join(occ_list) if occ_list else ""
            if occ_text:
                st.markdown(f"- [{file_name} ({occ_text})]({source})")
            else:
                st.markdown(f"- [{file_name}]({source})")
    else:
        st.markdown("*Nessuna fonte disponibile.*")

# ==============================
# SEZIONE 2: Agent Conversazionale ElevenLabs (centrato e ingrandito)
# ==============================
st.subheader("Agent Conversazionale ElevenLabs")

widget_html = """
<style>
  /* Contenitore centrante */
  .center-container {
      display: flex;
      justify-content: center;
      align-items: center;
      width: 100%;
      margin-top: 50px;
  }
  /* Dimensioni personalizzate del widget */
  .custom-widget {
      width: 800px !important;
      height: 600px !important;
  }
  /* Forza il widget a rispettare le dimensioni del contenitore */
  elevenlabs-convai {
      position: static !important;
      width: 100% !important;
      height: 100% !important;
      display: block !important;
  }
</style>
<div class="center-container">
  <div class="custom-widget">
    <elevenlabs-convai agent-id="nUnSSapc73VFkrd3Z73U"></elevenlabs-convai>
  </div>
</div>
<script src="https://elevenlabs.io/convai-widget/index.js" async type="text/javascript"></script>
"""

# Imposta l'altezza del componente HTML in base alle dimensioni definite
components.html(widget_html, height=700, scrolling=True)
