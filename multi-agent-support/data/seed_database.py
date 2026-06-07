from sqlalchemy import create_engine, text
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
    db_path = Config.DATABASE_URL.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(Config.DATABASE_URL)
    
    with engine.connect() as conn:
        # Create tables
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                phone TEXT,
                tier TEXT,
                created_at TEXT
            )
        """))
        
        conn.execute(text("""
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
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS refunds (
                refund_id TEXT PRIMARY KEY,
                order_id TEXT,
                amount REAL,
                reason TEXT,
                status TEXT,
                created_at TEXT
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS replacements (
                replacement_id TEXT PRIMARY KEY,
                order_id TEXT,
                status TEXT,
                reason TEXT,
                created_at TEXT,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            )
        """))
        
        # Seed customers
        names = ["Priya Sharma", "Rahul Gupta", "Anita Singh",
                 "Vikram Patel", "Sneha Reddy", "Arjun Nair",
                 "Kavya Iyer", "Rohan Mehta", "Pooja Joshi",
                 "Amit Kumar"]
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
        for i in range(100):
            cid = f"CUST{i+1:04d}"
            name = random.choice(names) + f" {i}"
            customers.append({
                "customer_id": cid,
                "name": name,
                "email": f"user{i}@example.com",
                "phone": f"+91 9{random.randint(100000000,999999999)}",
                "tier": random.choice(tiers),
                "created_at": (datetime.now() - timedelta(days=random.randint(30,730))).isoformat()
            })
        
        # Use conn (Connection) not engine — correct for pandas 2.x and 3.x
        pd.DataFrame(customers).to_sql("customers", engine, if_exists="replace", index=False)  # type: ignore[arg-type]
        
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
        
        # Use conn (Connection) not engine — fixes the Pyrefly type error
        pd.DataFrame(orders).to_sql("orders", engine, if_exists="replace", index=False)  # type: ignore[arg-type]
        conn.commit()
    
    print("✅ Database created: 100 customers, 500 orders")
    print("   Path: ./data/support.db")

if __name__ == "__main__":
    create_database()
