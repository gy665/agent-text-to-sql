import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.documents import Document

# Configuration de la page
st.set_page_config(page_title="Agent SQL AI", page_icon="🤖")
st.title("🤖 Chat avec ta Base de Données")

# 1. Setup
load_dotenv()

TABLE_DESCRIPTIONS = [
    "Table 'clients': Contient les infos utilisateurs (nom, email, pays, type d'abonnement VIP/Free).",
    "Table 'sales': Contient l'historique des ventes (date, montant, catégorie produit, id client)."
]
#chercheur de Tables(scout)
@st.cache_resource
def get_table_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(page_content=desc, metadata={"table_name": desc.split("'")[1]}) for desc in TABLE_DESCRIPTIONS]
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

# 3. La "Mémoire" (Few-Shot Examples) - Comme avant
@st.cache_resource
def get_few_shot_selector():
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

#initialisation des ressources 

table_retriever = get_table_retriever()
example_selector = get_few_shot_selector()






# 2. Interface de Chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Bonjour ! Pose-moi une question sur tes ventes ou tes clients."}]

# Afficher l'historique
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

    
# Zone de saisie utilisateur
user_query = st.chat_input("Ex: Quel est le client qui a le plus acheté ?")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.spinner("🧠 Réflexion en cours..."):
        
        # 1. SCOUT : Trouver les tables
        found_docs = table_retriever.similarity_search(user_query, k=2)
        selected_tables = []
        for doc in found_docs:
            table = doc.metadata.get('table_name')
            if table: selected_tables.append(table)
        selected_tables = list(set(selected_tables))
        
        # Sécurité : Si aucune table trouvée, on met 'sales' par défaut ou on arrête
        if not selected_tables:
            selected_tables = ['sales', 'clients']

        # 2. CONNEXION RESTREINTE
        db = SQLDatabase.from_uri("sqlite:///sales.db", include_tables=selected_tables)
        
        # 3. EXTRACTION DU SCHÉMA RÉEL (La "Magic Touch" de la V5 intégrée dans la V3)
        # On récupère le CREATE TABLE précis pour les tables sélectionnées
        real_schema_info = db.get_table_info(selected_tables)

        # 4. FEW-SHOT
        related_examples = example_selector.select_examples({"input": user_query})
        examples_str = "\n".join([f"- User: {ex['input']}\n  SQL: {ex['query']}" for ex in related_examples])

        # 5. PROMPT D'INGÉNIEUR
        # On combine : Instructions + Schéma Réel (DDL) + Exemples
        system_prompt = f"""
        Tu es un expert SQL.
        
        --- INFO CRITIQUE : SCHÉMA DE LA BASE ---
        Voici la définition exacte des tables que tu dois utiliser.
        N'utilise JAMAIS une colonne qui n'est pas écrite ici :
        
        {real_schema_info}
        
        --- EXEMPLES PERTINENTS ---
        {examples_str}
        
        --- INSTRUCTIONS ---
        1. Si l'utilisateur demande "articles", utilise la colonne texte (ex: product_category) vue dans le schéma ci-dessus.
        2. Ne fais pas de suppositions. Lis le schéma ci-dessus.
        3. Si la question est hors sujet, dis "Je ne sais pas".
        """
        try:
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            
            agent_executor = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="tool-calling",
                handle_parsing_errors=True,
                # On injecte le contexte riche ici
                prefix=system_prompt

                
            )
            
            response = agent_executor.invoke({"input": user_query})
            result = response['output']

           
            
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.chat_message("assistant").write(result)

            
            
            # Debug pour comprendre ce qui s'est passé
            with st.expander("🛠️ Voir le Cerveau (Debug)"):
                st.write(f"**Tables choisies :** {selected_tables}")
                st.write("**Schéma injecté (La sécurité) :**")
                st.code(real_schema_info, language="sql")
                st.write("**Exemples utilisés :**")
                st.text(examples_str)

        except Exception as e:
            st.error(f"Oups : {e}")