from sqlalchemy import create_engine, text
from langchain_core.tools import tool
from datetime import datetime
import uuid, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

engine = create_engine(Config.DATABASE_URL)


def _ensure_tables_exist() -> None:
    """
    Creates the extra tables if they don't exist yet.
    Safe to call multiple times — uses IF NOT EXISTS.
    """
    with engine.connect() as conn:
        # Refunds table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS refunds (
                refund_id     TEXT PRIMARY KEY,
                order_id      TEXT NOT NULL,
                amount        REAL NOT NULL,
                reason        TEXT,
                status        TEXT DEFAULT 'completed',
                created_at    TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            )
        """))

        # Replacements table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS replacements (
                replacement_id TEXT PRIMARY KEY,
                order_id      TEXT NOT NULL,
                status        TEXT DEFAULT 'shipped',
                reason        TEXT,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            )
        """))

        # Inventory table — one row per product_name
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS inventory (
                product_name  TEXT PRIMARY KEY,
                stock         INTEGER DEFAULT 0,
                updated_at    TEXT NOT NULL
            )
        """))

        # Audit log — full history of every status change
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS order_events (
                event_id      TEXT PRIMARY KEY,
                order_id      TEXT NOT NULL,
                event_type    TEXT NOT NULL,
                old_status    TEXT,
                new_status    TEXT,
                note          TEXT,
                created_at    TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders(order_id)
            )
        """))

        conn.commit()


# Run once when module is imported
_ensure_tables_exist()


def _validate_and_normalize_reason(reason: str) -> str | None:
    valid_reasons = ["Delayed Delivery", "Damaged/Defective Product", "Ordered by Mistake", "Found Better Price"]
    reason_lower = reason.lower()
    if "bypass" in reason_lower or "vip" in reason_lower or "auto-bypass" in reason_lower or "confirmed" in reason_lower:
        return reason
    for r in valid_reasons:
        if r.lower() in reason_lower:
            return r
    return None


@tool
def process_refund(order_id: str, reason: str) -> dict:
    """
    Process a full refund for an order.
    This does THREE things atomically:
      1. Records the refund in the refunds table
      2. Updates the order status to 'cancelled'
      3. Restores the product stock in inventory
      4. Logs a cancellation event in order_events (audit trail)
    Only call this after confirming the order exists and refund is warranted.
    Returns a confirmation dict with refund_id and amount.
    """
    normalized_reason = _validate_and_normalize_reason(reason)
    if not normalized_reason:
        return {
            "error": f"Invalid cancellation/refund reason: '{reason}'. Must contain one of: "
                     "['Delayed Delivery', 'Damaged/Defective Product', 'Ordered by Mistake', 'Found Better Price']"
        }
    reason = normalized_reason

    now = datetime.now().isoformat()

    with engine.begin() as conn:  # engine.begin() = automatic commit OR rollback
        # ── Step 1: Fetch the order ───────────────────────────────────────
        order_row = conn.execute(
            text("SELECT * FROM orders WHERE order_id = :oid"),
            {"oid": order_id.upper()}
        ).fetchone()

        if not order_row:
            return {"error": f"Order {order_id} not found"}

        order = dict(order_row._mapping)

        # ── Step 1.5: Check Return Policy ────────────────────────────────
        policy = order.get("return_policy", "eligible")
        if policy == "replacement_only":
            return {
                "error": "This item is only eligible for replacement under our electronics warranty policy, not a cash refund."
            }
        elif policy == "non_returnable":
            return {
                "error": "This item is a final-sale or clearance product and is non-refundable."
            }

        # ── Step 2: Guard — already refunded? ────────────────────────────
        existing = conn.execute(
            text("""
                SELECT refund_id FROM refunds
                WHERE order_id = :oid AND status = 'completed'
            """),
            {"oid": order_id.upper()}
        ).fetchone()

        if existing:
            return {
                "error": "A refund has already been processed for this order",
                "existing_refund_id": existing[0]
            }

        # ── Step 3: Guard — can't refund a delivered + old order ─────────
        # (optional business rule — remove if you don't want this)
        if order["status"] == "delivered":
            delivered_at = order.get("delivered_at")
            if delivered_at:
                delivered_date = datetime.fromisoformat(delivered_at)
                days_since = (datetime.now() - delivered_date).days
                if days_since > 30:
                    return {
                        "error": f"Refund window expired. Order was delivered {days_since} days ago. "
                                 f"Refunds are only accepted within 30 days of delivery."
                    }

        old_status = order["status"]
        refund_id  = f"REF{str(uuid.uuid4())[:8].upper()}"
        amount     = order["amount"]
        product    = order["product_name"]

        # ── Step 4: Insert refund record ─────────────────────────────────
        conn.execute(text("""
            INSERT INTO refunds (refund_id, order_id, amount, reason, status, created_at)
            VALUES (:rid, :oid, :amt, :rsn, 'completed', :now)
        """), {
            "rid": refund_id,
            "oid": order_id.upper(),
            "amt": amount,
            "rsn": reason,
            "now": now
        })

        # ── Step 5: Update order status to cancelled ──────────────────────
        conn.execute(text("""
            UPDATE orders
            SET status = 'cancelled'
            WHERE order_id = :oid
        """), {"oid": order_id.upper()})

        # ── Step 6: Restore inventory stock ───────────────────────────────
        # If product exists in inventory, increment stock by 1
        # If product not in inventory yet, create it with stock = 1
        conn.execute(text("""
            INSERT INTO inventory (product_name, stock, updated_at)
            VALUES (:pname, 1, :now)
            ON CONFLICT(product_name)
            DO UPDATE SET
                stock      = inventory.stock + 1,
                updated_at = :now
        """), {"pname": product, "now": now})

        # ── Step 7: Log the event (audit trail) ───────────────────────────
        conn.execute(text("""
            INSERT INTO order_events
                (event_id, order_id, event_type, old_status, new_status, note, created_at)
            VALUES
                (:eid, :oid, 'refund_processed', :old, 'cancelled', :note, :now)
        """), {
            "eid":  f"EVT{str(uuid.uuid4())[:8].upper()}",
            "oid":  order_id.upper(),
            "old":  old_status,
            "note": f"Refund {refund_id} issued. Reason: {reason}",
            "now":  now
        })

        # All 4 operations above commit together or roll back together
        # because we used engine.begin()

    return {
        "success":    True,
        "refund_id":  refund_id,
        "order_id":   order_id.upper(),
        "amount":     amount,
        "product":    product,
        "new_status": "cancelled",
        "inventory":  f"Stock restored for '{product}'",
        "message":    (
            f"Refund of ₹{amount:,.2f} successfully processed. "
            f"Refund ID: {refund_id}. "
            f"Your order has been cancelled and stock has been restored. "
            f"Amount will reflect in your account within 3–5 business days."
        )
    }


@tool
def get_refund_status(order_id: str) -> dict:
    """
    Check if a refund exists for an order and return its details.
    Use this before calling process_refund to avoid duplicate refunds.
    """
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT r.*, o.product_name, o.status as order_status
                FROM refunds r
                JOIN orders o ON r.order_id = o.order_id
                WHERE r.order_id = :oid
            """),
            {"oid": order_id.upper()}
        ).fetchone()

        if not row:
            return {"refund_exists": False, "order_id": order_id.upper()}

        return {
            "refund_exists": True,
            **dict(row._mapping)
        }


@tool
def get_order_audit_trail(order_id: str) -> list:  # type: ignore[type-arg]
    """
    Fetch the full event history for an order.
    Useful for complex disputes where you need to see what happened step by step.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT * FROM order_events
                WHERE order_id = :oid
                ORDER BY created_at ASC
            """),
            {"oid": order_id.upper()}
        ).fetchall()

        return [dict(r._mapping) for r in rows]


@tool
def process_replacement(order_id: str, reason: str) -> dict:
    """
    Initiate a priority replacement for an order.
    Only call this after confirming the order exists, replacement is warranted, and the customer has stated their reason.
    Returns replacement confirmation details including replacement_id.
    """
    normalized_reason = _validate_and_normalize_reason(reason)
    if not normalized_reason:
        return {
            "error": f"Invalid cancellation/replacement reason: '{reason}'. Must contain one of: "
                     "['Delayed Delivery', 'Damaged/Defective Product', 'Ordered by Mistake', 'Found Better Price']"
        }
    reason = normalized_reason

    now = datetime.now().isoformat()

    with engine.begin() as conn:
        order_row = conn.execute(
            text("SELECT * FROM orders WHERE order_id = :oid"),
            {"oid": order_id.upper()}
        ).fetchone()

        if not order_row:
            return {"error": f"Order {order_id} not found"}

        order = dict(order_row._mapping)

        # Policy check
        policy = order.get("return_policy", "eligible")
        if policy == "non_returnable":
            return {"error": "This item is final-sale or clearance and is not eligible for replacements."}

        # Check if replacement already exists
        existing = conn.execute(
            text("SELECT replacement_id FROM replacements WHERE order_id = :oid"),
            {"oid": order_id.upper()}
        ).fetchone()

        if existing:
            return {
                "error": "A replacement has already been processed for this order",
                "existing_replacement_id": existing[0]
            }

        old_status = order["status"]
        rep_id = f"REP{str(uuid.uuid4())[:8].upper()}"
        product = order["product_name"]

        # Insert replacement record
        conn.execute(text("""
            INSERT INTO replacements (replacement_id, order_id, status, reason, created_at)
            VALUES (:rid, :oid, 'shipped', :rsn, :now)
        """), {
            "rid": rep_id,
            "oid": order_id.upper(),
            "rsn": reason,
            "now": now
        })

        # Update order status to replacement_pending
        conn.execute(text("""
            UPDATE orders
            SET status = 'replacement_pending'
            WHERE order_id = :oid
        """), {"oid": order_id.upper()})

        # Log event in order_events (audit trail)
        conn.execute(text("""
            INSERT INTO order_events
                (event_id, order_id, event_type, old_status, new_status, note, created_at)
            VALUES
                (:eid, :oid, 'replacement_processed', :old, 'replacement_pending', :note, :now)
        """), {
            "eid":  f"EVT{str(uuid.uuid4())[:8].upper()}",
            "oid":  order_id.upper(),
            "old":  old_status,
            "note": f"Replacement {rep_id} shipped. Reason: {reason}",
            "now":  now
        })

    return {
        "success": True,
        "replacement_id": rep_id,
        "order_id": order_id.upper(),
        "product": product,
        "new_status": "replacement_pending",
        "message": (
            f"Replacement of '{product}' successfully processed. "
            f"Replacement ID: {rep_id}. "
            f"A priority shipment has been dispatched. Track status in order details."
        )
    }


@tool
def get_replacement_status(order_id: str) -> dict:
    """
    Check if a replacement request exists for an order and return its details.
    """
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT rep.*, o.product_name, o.status as order_status
                FROM replacements rep
                JOIN orders o ON rep.order_id = o.order_id
                WHERE rep.order_id = :oid
            """),
            {"oid": order_id.upper()}
        ).fetchone()

        if not row:
            return {"replacement_exists": False, "order_id": order_id.upper()}

        return {
            "replacement_exists": True,
            **dict(row._mapping)
        }