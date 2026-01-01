import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent

#load api key
load_dotenv()


#connexion de db


db_uri = "sqlite:///sales.db"
db = SQLDatabase.from_uri(db_uri)


llm = ChatGroq(

    model = "llama-3.3-70b-versatile",
    temperature = 0,
    verbose = True    
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)


#create agent 
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type= "tool-calling",
    handle_parsing_error= True
)




# test

def poser_question(question):
    print(f"\n🤖 Question : {question}")
    print("-" * 50)
    try:
        response = agent_executor.invoke(question)
        print("-" * 50)
        print(f"✅ Réponse Finale : {response['output']}")
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    # Test 1 : Simple comptage
    poser_question("Combien de clients avons-nous au total ?")

    # Test 2 : Jointure (Clients + Ventes)
    poser_question("Quel est le montant total des ventes pour les clients en France ?")