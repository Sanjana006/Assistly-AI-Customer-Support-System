from sqlalchemy import create_engine, text
from langchain_core.tools import tool
from typing import Optional
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

engine = create_engine(Config.DATABASE_URL)

@tool
def get_order_by_id(order_id: str) -> dict:
    """
    Fetch a specific order from the database by order ID.
    Use this when a customer mentions an order number.
    Returns order details including status, product, amount, delivery dates.
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT o.*, c.name as customer_name, c.email, c.tier
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE o.order_id = :oid
            """),
            {"oid": order_id.upper()}
        ).fetchone()
        
        if not result:
            return {"error": f"Order {order_id} not found"}
        
        return dict(result._mapping)

@tool
def get_orders_by_customer_email(email: str) -> list:
    """
    Get all orders for a customer by their email address.
    Use when customer asks about 'my orders' without a specific ID.
    """
    with engine.connect() as conn:
        results = conn.execute(
            text("""
                SELECT o.order_id, o.product_name, o.status, 
                       o.amount, o.created_at, o.expected_delivery
                FROM orders o
                JOIN customers c ON o.customer_id = c.customer_id
                WHERE c.email = :email
                ORDER BY o.created_at DESC
                LIMIT 5
            """),
            {"email": email}
        ).fetchall()
        
        return [dict(r._mapping) for r in results]

@tool
def process_refund(order_id: str, reason: str) -> dict:
    """
    Process a refund for an order.
    Only call this after confirming the order exists and refund is warranted.
    Returns refund confirmation with refund_id.
    """
    import uuid
    from datetime import datetime
    
    with engine.connect() as conn:
        order = conn.execute(
            text("SELECT * FROM orders WHERE order_id = :oid"),
            {"oid": order_id.upper()}
        ).fetchone()
        
        if not order:
            return {"error": f"Order {order_id} not found"}
        
        order_dict = dict(order._mapping)
        
        # Check if refund already exists
        existing = conn.execute(
            text("SELECT * FROM refunds WHERE order_id = :oid AND status='completed'"),
            {"oid": order_id.upper()}
        ).fetchone()
        
        if existing:
            return {"error": "Refund already processed for this order"}
        
        refund_id = f"REF{str(uuid.uuid4())[:8].upper()}"
        
        conn.execute(text("""
            INSERT INTO refunds (refund_id, order_id, amount, reason, status, created_at)
            VALUES (:rid, :oid, :amt, :rsn, 'completed', :now)
        """), {
            "rid": refund_id,
            "oid": order_id.upper(),
            "amt": order_dict["amount"],
            "rsn": reason,
            "now": datetime.now().isoformat()
        })
        conn.commit()
        
        return {
            "success": True,
            "refund_id": refund_id,
            "amount": order_dict["amount"],
            "order_id": order_id,
            "message": f"Refund of ₹{order_dict['amount']:.2f} processed. "
                        f"Will reflect in 3-5 business days."
        }