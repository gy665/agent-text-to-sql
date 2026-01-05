import streamlit as st
import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv, find_dotenv
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

# ==========================================
# 1. SETUP CLÉ API (BLINDÉ)
# ==========================================
load_dotenv(find_dotenv())

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        pass

# On force la variable d'environnement pour que la librairie ChatGroq soit contente
if api_key:
    os.environ["GROQ_API_KEY"] = api_key
else:
    st.error("🚨 Clé API introuvable ! Configurez GROQ_API_KEY dans .env ou les Secrets Streamlit.")
    st.stop()

# ==========================================
# 2. GÉNÉRATION AUTOMATIQUE DB (TA SOLUTION)
# ==========================================
# Si le fichier n'existe pas (cas du Cloud), on le crée à la volée !
if not os.path.exists("sales.db"):
    with st.spinner("🛠️ Initialisation de la base de données sur le Cloud..."):
        conn = sqlite3.connect('sales.db')
        cursor = conn.cursor()
        
        # Création Tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            client_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            country TEXT NOT NULL,
            subscription_type TEXT CHECK(subscription_type IN ('Free', 'Premium', 'VIP'))
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sales (
            sale_id INTEGER PRIMARY KEY,
            client_id INTEGER,
            sale_date DATE NOT NULL,
            amount DECIMAL(10, 2) NOT NULL,
            product_category TEXT NOT NULL,
            FOREIGN KEY (client_id) REFERENCES clients (client_id)
        )
        ''')
        
        # Insertion Données
        clients_data = [
            (1, 'Alice Dupont', 'alice@example.com', 'France', 'VIP'),
            (2, 'Bob Martin', 'bob@example.com', 'Canada', 'Premium'),
            (3, 'Charlie Smith', 'charlie@example.com', 'USA', 'Free'),
            (4, 'David Lee', 'david@example.com', 'France', 'Premium'),
            (5, 'Eve Tran', 'eve@example.com', 'Vietnam', 'VIP')
        ]
        cursor.executemany('INSERT OR IGNORE INTO clients VALUES (?,?,?,?,?)', clients_data)
        
        sales_data = [
            (101, 1, '2023-01-15', 150.00, 'Electronics'),
            (102, 1, '2023-02-10', 300.50, 'Books'),
            (103, 2, '2023-03-05', 1200.00, 'Furniture'),
            (104, 3, '2023-01-20', 25.00, 'Books'),
            (105, 4, '2023-04-12', 450.00, 'Electronics'),
            (106, 5, '2023-05-30', 900.00, 'Electronics'),
            (107, 1, '2023-06-01', 50.00, 'Accessories')
        ]
        cursor.executemany('INSERT OR IGNORE INTO sales VALUES (?,?,?,?,?)', sales_data)
        conn.commit()
        conn.close()
        st.success("✅ Base de données générée avec succès !")

# ==========================================
# 3. CONFIGURATION AGENT & RAG
# ==========================================

TABLE_DESCRIPTIONS = [
    "Table 'clients': Contient les infos utilisateurs (nom, email, pays, type d'abonnement VIP/Free).",
    "Table 'sales': Contient l'historique des ventes (date, montant, catégorie produit, id client)."
]

@st.cache_resource
def get_table_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [Document(page_content=desc, metadata={"table_name": desc.split("'")[1]}) for desc in TABLE_DESCRIPTIONS]
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

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

table_retriever = get_table_retriever()
example_selector = get_few_shot_selector()

# ==========================================
# 4. INTERFACE CHAT
# ==========================================

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Bonjour ! Je suis connecté à la base de données. Pose ta question."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    if "chart_data" in msg:
        st.bar_chart(msg["chart_data"])
    
user_query = st.chat_input("Ex: Quel est le client qui a le plus acheté ?")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.spinner("🧠 Réflexion en cours..."):
        
        # SCOUT
        found_docs = table_retriever.similarity_search(user_query, k=2)
        selected_tables = list(set([doc.metadata.get('table_name') for doc in found_docs if doc.metadata.get('table_name')]))
        if not selected_tables: selected_tables = ['sales', 'clients']

        # CONNEXION
        db = SQLDatabase.from_uri("sqlite:///sales.db", include_tables=selected_tables)
        real_schema_info = db.get_table_info(selected_tables)

        # FEW-SHOT
        related_examples = example_selector.select_examples({"input": user_query})
        examples_str = "\n".join([f"- User: {ex.get('input','')}\n  SQL: {ex.get('query','')}" for ex in related_examples])

        # PROMPT
        system_prompt = f"""
        Tu es un expert SQL.
        SCHEMA: {real_schema_info}
        EXEMPLES: {examples_str}
        INSTRUCTIONS:
        1. Utilise le schéma ci-dessus.
        2. Si on demande "articles", utilise 'product_category'.
        3. Réponds uniquement avec du SQL ou du texte explicatif si hors-sujet.
        """
        
        try:
            # ON NE PASSE PLUS api_key=... CAR C'EST DANS OS.ENVIRON
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)
            
            agent_executor = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="tool-calling",
                handle_parsing_errors=True,
                prefix=system_prompt,
                agent_executor_kwargs={"return_intermediate_steps": True}
            )
            
            response = agent_executor.invoke({"input": user_query})
            result = response['output']

            st.session_state.messages.append({"role": "assistant", "content": result})
            st.chat_message("assistant").write(result)

            # GRAPHIQUE
            chart_data = None
            sql_query = None
            if "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    if step[0].tool == "sql_db_query":
                        sql_query = step[0].tool_input
            
            if sql_query:
                if isinstance(sql_query, dict): sql_query = sql_query.get('query', str(sql_query))
                sql_query = str(sql_query).replace("```sql", "").replace("```", "").strip()
                try:
                    df = pd.read_sql(sql_query, db._engine)
                    if len(df) > 1 and len(df.columns) >= 2:
                        if df.dtypes[0] == 'object' or df.dtypes[0] == 'string':
                            df = df.set_index(df.columns[0])
                        chart_data = df
                except: pass

            if chart_data is not None:
                st.bar_chart(chart_data)
                st.session_state.messages[-1]["chart_data"] = chart_data

            with st.expander("🛠️ Debug"):
                st.write(f"Tables: {selected_tables}")
                st.code(real_schema_info)

        except Exception as e:
            st.error(f"Erreur : {e}")