import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Vérification de sécurité
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ Erreur : La clé GROQ_API_KEY est introuvable.")
    exit()

try:
    print("🤖 Initialisation du modèle Llama 3.3 (via Groq)...")
    
    # MISE A JOUR DU MODELE ICI
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", # <-- C'est le nouveau modèle actif
        temperature=0
    )
    
    print("📨 Envoi de la requête de test...")
    response = llm.invoke("Réponds juste par 'OK' si tu me reçois.")
    
    print(f"✅ Réponse de l'IA : {response.content}")
    print("🎉 L'environnement est prêt !")

except Exception as e:
    print(f"❌ Une erreur est survenue : {e}")