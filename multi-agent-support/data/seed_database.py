import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta
import os
import sys

# Ensure config is importable relative to this file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def create_database():
    random.seed(42)
    if Config.DATABASE_URL.startswith("sqlite://"):
        from sqlalchemy.engine import make_url
        db_path = make_url(Config.DATABASE_URL).database or "support.db"
    else:
        db_path = Config.DATABASE_URL.replace("sqlite:///", "") or "support.db"
    if os.path.dirname(db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Use raw sqlite3 connection directly to bypass SQLAlchemy / pandas version mismatch issues
    conn = sqlite3.connect(db_path)
    try:
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                phone TEXT,
                tier TEXT,
                created_at TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT,
                product_name TEXT,
                amount REAL,
                status TEXT,
                created_at TEXT,
                expected_delivery TEXT,
                delivered_at TEXT,
                tracking_number TEXT,
                return_policy TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS refunds (
                refund_id TEXT PRIMARY KEY,
                order_id TEXT,
                amount REAL,
                reason TEXT,
                status TEXT,
                created_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS replacements (
                replacement_id TEXT PRIMARY KEY,
                order_id TEXT,
                status TEXT,
                reason TEXT,
                created_at TEXT,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            )
        """)
        
        # Seed customers
        first_names = [
            "Emma", "Liam", "Olivia", "Noah", "Ava", "Oliver", "Sophia", "Lucas", "Mia", "Alexander",
            "Isabella", "Ethan", "Charlotte", "Mason", "Amelia", "James", "Harper", "Benjamin", "Evelyn", "Daniel",
            "Elijah", "Logan", "Grace", "Caleb", "Zoe", "Jackson", "Lily", "Jacob", "Chloe", "Michael",
            "Aarav", "Aditi", "Aditya", "Ananya", "Ankit", "Dev", "Divya", "Gaurav", "Ishaan", "Karan",
            "Karthik", "Kavita", "Kiran", "Manish", "Meera", "Neha", "Nikhil", "Nisha", "Pranav", "Ravi",
            "Riya", "Sandeep", "Sanjana", "Shreya", "Siddharth", "Sumit", "Swati", "Varun", "Vikas", "Yash",
            "Deep", "Raj", "Rani", "Zoya", "Kavya", "Aanya", "Abhishek", "Aishwarya", "Alok", "Anil"
        ]
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
            "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
            "Sen", "Roy", "Das", "Banerjee", "Chatterjee", "Mukherjee", "Bose", "Choudhury", "Dutta", "Mitra",
            "Rao", "Naidu", "Pillai", "Menon", "Krishnan", "Deshmukh", "Kulkarni", "Patil", "Bapat", "Gokhale",
            "Sardesai", "Pawar", "Shinde", "Verma", "Srivastava", "Tripathi", "Pandey", "Mishra", "Tiwari", "Dubey"
        ]
        tiers = ["silver", "gold", "platinum"]
        products = [
            "Samsung Galaxy S24", "iPhone 15 Pro", "Noise Smartwatch",
            "boAt Earbuds", "Xiaomi Laptop", "HP Printer",
            "Sony Headphones", "Kindle Paperwhite",
            "Canon Camera", "JBL Speaker"
        ]
        statuses = ["delivered", "shipped", "processing",
                    "delayed", "cancelled"]
        
        customers = []
        used_names = set()
        for i in range(100):
            cid = f"CUST{i+1:04d}"
            while True:
                name = f"{random.choice(first_names)} {random.choice(last_names)}"
                if name not in used_names:
                    used_names.add(name)
                    break
            email_prefix = name.lower().replace(" ", "")
            customers.append({
                "customer_id": cid,
                "name": name,
                "email": f"{email_prefix}{i}@example.com",
                "phone": f"+91 9{random.randint(100000000,999999999)}",
                "tier": random.choice(tiers),
                "created_at": (datetime.now() - timedelta(days=random.randint(30,730))).isoformat()
            })
        
        pd.DataFrame(customers).to_sql("customers", conn, if_exists="replace", index=False)
        
        # Seed orders
        orders = []
        for i in range(500):
            cust = random.choice(customers)
            created = datetime.now() - timedelta(days=random.randint(1, 60))
            expected = created + timedelta(days=random.randint(3, 10))
            status = random.choice(statuses)
            delivered = (expected + timedelta(days=random.randint(0,3))).isoformat() \
                        if status == "delivered" else None
            
            product = random.choice(products)
            # Assign return policies based on product category
            if product in ["Kindle Paperwhite", "Noise Smartwatch", "Samsung Galaxy S24", "iPhone 15 Pro", "Xiaomi Laptop"]:
                policy = "replacement_only"
            elif product in ["boAt Earbuds", "Sony Headphones"]:
                policy = "non_returnable"
            else:
                policy = "eligible"

            orders.append({
                "order_id": f"ORD{i+1:05d}",
                "customer_id": cust["customer_id"],
                "product_name": product,
                "amount": round(random.uniform(299, 89999), 2),
                "status": status,
                "created_at": created.isoformat(),
                "expected_delivery": expected.isoformat(),
                "delivered_at": delivered,
                "tracking_number": f"IND{random.randint(100000000,999999999)}",
                "return_policy": policy
            })
        
        pd.DataFrame(orders).to_sql("orders", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()
    
    print("✅ Database created: 100 customers, 500 orders")
    print("   Path: ./data/support.db")

if __name__ == "__main__":
    create_database()
