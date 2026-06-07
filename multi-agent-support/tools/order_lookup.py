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

@tool
def get_order_journey_details(order_id: str) -> dict:
    """
    Fetch detailed, step-by-step logistics journey, warehouse origin, and delay reasoning for an order.
    Use this tool to explain the specific "why" behind any order delays or status updates.
    """
    order_info = get_order_by_id.invoke({"order_id": order_id})
    if "error" in order_info:
        return order_info
    
    oid = order_info["order_id"]
    status = order_info["status"]
    
    # Use a deterministic hash of the order ID to assign a warehouse, issue, and details
    # so that it remains consistent for each order.
    hash_val = sum(ord(c) for c in oid)
    
    warehouses = ["Mumbai (MH-01 Hub)", "Delhi NCR (DL-04 Fulfillment Center)", "Bengaluru (KA-03 Sort Facility)"]
    warehouse = warehouses[hash_val % len(warehouses)]
    
    if status == "delivered":
        return {
            "order_id": oid,
            "warehouse": warehouse,
            "status": "delivered",
            "reason": "Delivered successfully",
            "milestones": [
                {"time": "Day 1", "location": warehouse, "event": "Order packed and dispatched"},
                {"time": "Day 2", "location": "Local Delivery Hub", "event": "Out for delivery"},
                {"time": "Day 3", "location": "Customer Destination", "event": "Delivered and signed"}
            ]
        }
    elif status == "cancelled":
        return {
            "order_id": oid,
            "warehouse": warehouse,
            "status": "cancelled",
            "reason": "Cancelled by user or due to payment failure",
            "milestones": [
                {"time": "Day 1", "location": "Payment Gateway", "event": "Transaction flagged or cancelled by customer"}
            ]
        }
    elif status == "processing":
        return {
            "order_id": oid,
            "warehouse": warehouse,
            "status": "processing",
            "reason": "Undergoing packaging and quality checks",
            "milestones": [
                {"time": "Day 1", "location": warehouse, "event": "Order received and inventory allocated"}
            ]
        }
    else: # delayed or shipped
        issues = [
            {
                "category": "Stock Mismatch",
                "detail": f"A stock mismatch was detected during packaging for your {order_info['product_name']}. The warehouse is performing an emergency inventory audit.",
                "impact": "Dispatch held for 48 hours",
                "location": warehouse
            },
            {
                "category": "Logistics Flight Grounding",
                "detail": f"The cargo flight transporting the parcel from {warehouse} was grounded due to unexpected adverse weather conditions.",
                "impact": "Logistics transit delayed by 36 hours",
                "location": "Transit Air Hub"
            },
            {
                "category": "High Courier Backlog",
                "detail": "Local delivery courier services are experiencing a seasonal demand backlog, restricting daily dispatch capacity.",
                "impact": "Parcel sorting held at regional dispatch center",
                "location": "Regional Sorting Center"
            }
        ]
        issue = issues[hash_val % len(issues)]
        return {
            "order_id": oid,
            "warehouse": warehouse,
            "status": status,
            "category": issue["category"],
            "detail": issue["detail"],
            "impact": issue["impact"],
            "current_location": issue["location"],
            "milestones": [
                {"time": "Day 1", "location": warehouse, "event": "Order processed and packed"},
                {"time": "Day 2", "location": issue["location"], "event": f"Delay occurred: {issue['category']}"}
            ]
        }

@tool
def get_customer_incident_profile(email: str) -> dict:
    """
    Retrieve a customer's incident profile including their order history stats,
    number of delayed or cancelled orders, and completed refunds.
    Use this tool to evaluate customer frustration history and determine if they qualify
    for priority support, VIP escalation, or instant refund bypass.
    """
    with engine.connect() as conn:
        customer = conn.execute(
            text("SELECT customer_id, name, email, tier FROM customers WHERE email = :email"),
            {"email": email.strip()}
        ).fetchone()
        
        if not customer:
            return {"error": f"Customer with email {email} not found"}
        
        cust_id = customer.customer_id
        
        orders = conn.execute(
            text("SELECT order_id, status FROM orders WHERE customer_id = :cid"),
            {"cid": cust_id}
        ).fetchall()
        
        refunds = conn.execute(
            text("""
                SELECT r.refund_id 
                FROM refunds r
                JOIN orders o ON r.order_id = o.order_id
                WHERE o.customer_id = :cid
            """),
            {"cid": cust_id}
        ).fetchall()
        
        total_orders = len(orders)
        delayed_orders = sum(1 for o in orders if o.status == "delayed")
        cancelled_orders = sum(1 for o in orders if o.status == "cancelled")
        refunded_orders = len(refunds)
        
        return {
            "customer_id": cust_id,
            "customer_name": customer.name,
            "email": customer.email,
            "tier": customer.tier,
            "total_orders": total_orders,
            "delayed_orders": delayed_orders,
            "cancelled_orders": cancelled_orders,
            "refunded_orders": refunded_orders,
            "incident_count": delayed_orders + cancelled_orders
        }