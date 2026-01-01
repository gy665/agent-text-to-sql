import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate

# Configuration de la page
st.set_page_config(page_title="Agent SQL AI", page_icon="🤖")
st.title("🤖 Chat avec ta Base de Données")

# 1. Setup
load_dotenv()

# Mise en cache des ressources pour ne pas recharger à chaque clic (Rapidité)
@st.cache_resource
def get_engine():
    db_uri = "sqlite:///sales.db"
    return SQLDatabase.from_uri(db_uri)

@st.cache_resource
def get_vector_store():
    # Définition des exemples
    examples = [
        {"input": "Liste tous les clients VIP.", "query": "SELECT * FROM clients WHERE subscription_type = 'VIP';"},
        {"input": "Qui a dépensé le plus d'argent ?", "query": "SELECT t1.name, SUM(t2.amount) as total_spent FROM clients t1 JOIN sales t2 ON t1.client_id = t2.client_id GROUP BY t1.name ORDER BY total_spent DESC LIMIT 1;"},
        {"input": "Combien de ventes d'Electronics on a fait ?", "query": "SELECT COUNT(*) FROM sales WHERE product_category = 'Electronics';"},
        {"input": "Donne moi les emails des clients canadiens.", "query": "SELECT email FROM clients WHERE country = 'Canada';"}
    ]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    selector = SemanticSimilarityExampleSelector.from_examples(
        examples, embeddings, Chroma, k=2, input_keys=["input"]
    )
    return selector

db = get_engine()
example_selector = get_vector_store()

# Clé API Check
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Clé API manquante dans le fichier .env")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="tool-calling",
    handle_parsing_errors=True
)

# 2. Interface de Chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Bonjour ! Pose-moi une question sur tes ventes ou tes clients."}]

# Afficher l'historique
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Zone de saisie utilisateur
user_query = st.chat_input("Ex: Quel est le client qui a le plus acheté ?")

if user_query:
    # Afficher la question
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    # Logique RAG (Récupération des exemples)
    with st.spinner("L'IA réfléchit et consulte les exemples..."):
        related_examples = example_selector.select_examples({"input": user_query})
        examples_str = ""
        for ex in related_examples:
            examples_str += f"- Q: {ex['input']}\n  SQL: {ex['query']}\n"
        
        # Prompt enrichi
        prompt_final = f"""
        Tu es un expert SQL. Utilise ces exemples pour comprendre la structure :
        {examples_str}
        
        Réponds à la question suivante en exécutant une requête SQL :
        {user_query}
        """
        
        try:
            response = agent_executor.invoke(prompt_final)
            result = response['output']
            
            # Afficher la réponse
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.chat_message("assistant").write(result)
            
            # (Optionnel) Afficher les exemples utilisés pour le debug dans un "expander"
            with st.expander("Voir la logique interne (Debug)"):
                st.write("Exemples injectés :")
                st.text(examples_str)
                
        except Exception as e:
            st.error(f"Erreur : {e}")