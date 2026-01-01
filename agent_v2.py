import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

# 1. Setup de base
load_dotenv()

# Attention aux 3 slashs pour SQLite
db_uri = "sqlite:///sales.db"
db = SQLDatabase.from_uri(db_uri)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------------------------------------------------
# 2. La "Base de Connaissance" (Golden SQL)
# ---------------------------------------------------------
examples = [
    {
        "input": "Liste tous les clients VIP.",
        "query": "SELECT * FROM clients WHERE subscription_type = 'VIP';"
    },
    {
        "input": "Qui a dépensé le plus d'argent ?",
        "query": "SELECT t1.name, SUM(t2.amount) as total_spent FROM clients t1 JOIN sales t2 ON t1.client_id = t2.client_id GROUP BY t1.name ORDER BY total_spent DESC LIMIT 1;"
    },
    {
        "input": "Combien de ventes d'Electronics on a fait ?",
        "query": "SELECT COUNT(*) FROM sales WHERE product_category = 'Electronics';"
    },
    {
        "input": "Donne moi les emails des clients canadiens.",
        "query": "SELECT email FROM clients WHERE country = 'Canada';"
    }
]

print("🧠 Chargement du modèle d'Embeddings (déjà en cache normalement)...")
# Note: On ignore les warnings de dépréciation pour l'instant
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    Chroma,
    k=2,
    input_keys=["input"],
)

# ---------------------------------------------------------
# 3. L'Agent (Simplifié)
# ---------------------------------------------------------
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# On crée l'agent standard sans prompt compliqué pour l'instant
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="tool-calling",
    handle_parsing_errors=True
)

# ---------------------------------------------------------
# 4. La Fonction "Intelligente" (C'est ici que la magie opère)
# ---------------------------------------------------------
def poser_question_intelligente(question):
    print(f"\n🤖 Question Utilisateur : {question}")
    print("-" * 20)
    
    # ÉTAPE A : On cherche les exemples pertinents dans la base vectorielle
    # L'IA va comparer ta question avec les exemples stockés
    related_examples = example_selector.select_examples({"input": question})
    
    # ÉTAPE B : On construit un message contextuel
    # On formate les exemples trouvés en texte
    examples_str = ""
    for ex in related_examples:
        examples_str += f"- Question: {ex['input']}\n  SQL: {ex['query']}\n"
    
    print(f"💡 Exemples similaires trouvés et injectés :\n{examples_str}")
    print("-" * 20)

    # ÉTAPE C : On crée le Prompt Final
    # On injecte les exemples DANS la question pour guider Llama 3
    prompt_final = f"""
    Tu es un expert SQL.
    Voici des exemples de requêtes CORRECTES qui ressemblent à la demande actuelle. Utilise-les pour comprendre la structure de la base (noms de tables, jointures) :
    
    {examples_str}
    
    Maintenant, réponds à cette nouvelle demande en générant et exécutant le SQL :
    Demande : {question}
    """
    
    try:
        # ÉTAPE D : On lance l'agent avec ce prompt enrichi
        response = agent_executor.invoke(prompt_final)
        print(f"✅ RÉSULTAT FINAL : {response['output']}")
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    # Test : Une question qui nécessite de comprendre la logique "Volume d'achat"
    poser_question_intelligente("Quel est le client qui a le plus gros volume d'achat ?")