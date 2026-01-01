import sqlite3

def create_database():
    # Cela va créer un fichier 'sales.db' dans ton dossier
    conn = sqlite3.connect('sales.db')
    cursor = conn.cursor()

    print("🛠️ Création des tables...")

    # 1. Table Clients
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS clients (
        client_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT,
        country TEXT NOT NULL,
        subscription_type TEXT CHECK(subscription_type IN ('Free', 'Premium', 'VIP'))
    )
    ''')

    # 2. Table Ventes (Sales)
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

    print("📥 Insertion des données de test...")

    # Données Clients
    clients_data = [
        (1, 'Alice Dupont', 'alice@example.com', 'France', 'VIP'),
        (2, 'Bob Martin', 'bob@example.com', 'Canada', 'Premium'),
        (3, 'Charlie Smith', 'charlie@example.com', 'USA', 'Free'),
        (4, 'David Lee', 'david@example.com', 'France', 'Premium'),
        (5, 'Eve Tran', 'eve@example.com', 'Vietnam', 'VIP')
    ]
    cursor.executemany('INSERT OR IGNORE INTO clients VALUES (?,?,?,?,?)', clients_data)

    # Données Ventes
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
    print("✅ Base de données 'sales.db' créée avec succès !")

if __name__ == "__main__":
    create_database()