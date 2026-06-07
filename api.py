from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agents.orchestrator import process_ticket
import uvicorn

app = FastAPI(
    title="Multi-Agent Support System",
    description="AI-powered customer support with specialist agents",
    version="1.0.0"
)

class TicketRequest(BaseModel):
    message: str
    conversation_history: Optional[list] = []
    customer_email: Optional[str] = None

class TicketResponse(BaseModel):
    ticket_id: str
    final_response: str
    intent: str
    sentiment: str
    frustration_score: float
    needs_escalation: bool
    escalation_reasons: list
    tools_called: list
    quality_score: float
    processing_time_ms: float

@app.post("/ticket", response_model=TicketResponse)
async def submit_ticket(request: TicketRequest):
    """Process a customer support ticket through the multi-agent pipeline."""
    try:
        result = process_ticket(
            user_message=request.message,
            conversation_history=request.conversation_history
        )
        return TicketResponse(**{k: result.get(k, v) 
                               for k, v in TicketResponse.model_fields.items()
                               if k in result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "agents": ["classifier", "resolver", "qa", "escalation"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)