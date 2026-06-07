from langchain_groq import ChatGroq          # ← changed to ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from pydantic import SecretStr
import json, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from tools.order_lookup import get_order_by_id, get_orders_by_customer_email, process_refund
from tools.kb_search import search_knowledge_base

# All tools the resolver can use
RESOLVER_TOOLS = [get_order_by_id, get_orders_by_customer_email, 
                  process_refund, search_knowledge_base]

def resolve_ticket(state: dict) -> dict:
    """
    Resolver Agent — the main worker.
    
    This uses ReAct pattern (Reason + Act):
    1. LLM reads the message + classification
    2. LLM decides: do I need to call a tool?
    3. If yes → calls tool → reads result → decides again
    4. Repeats until it has enough info to answer
    5. Generates final personalized response
    """
    
    classification = state.get("classification", {})
    
    system_prompt = f"""You are a helpful customer support agent for QuickShop India, an e-commerce platform.

Customer context:
- Intent: {classification.get('intent', 'unknown')}
- Sentiment: {classification.get('sentiment', 'neutral')}
- Urgency: {classification.get('urgency', 'medium')}
- Order ID mentioned: {classification.get('order_id', 'None')}

Your job:
1. Use tools to fetch relevant information (order status, knowledge base)
2. Generate a helpful, empathetic response based on REAL data
3. If processing a refund, confirm with process_refund tool
4. Match tone to sentiment — be extra warm/apologetic for angry customers
5. Be specific — use actual order details, actual dates, actual amounts

IMPORTANT RULES:
- Always fetch order data before mentioning order status
- Never make up order details — only use data from tools
- For angry customers, acknowledge frustration FIRST before solving
- Keep responses concise (2-4 sentences for simple issues)
- Sign off as "QuickShop Support Team"

Current conversation history included below."""

    api_key = SecretStr(Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None
    llm = ChatGroq(
        model=Config.get_model(),
        api_key=api_key,
        max_tokens=1000
    )
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(RESOLVER_TOOLS)
    
    # Build message history
    messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
    
    # Add conversation history if exists
    for msg in state.get("conversation_history", []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    messages.append(HumanMessage(content=state["user_message"]))
    
    # ReAct loop — keep going until LLM stops calling tools
    tool_registry = {t.name: t for t in RESOLVER_TOOLS}
    tools_called = []
    
    response = AIMessage(content="")
    for _ in range(Config.MAX_RETRIES):
        res = llm_with_tools.invoke(messages)
        if not isinstance(res, AIMessage):
            raise TypeError("Expected AIMessage from LLM")
        response = res
        messages.append(response)
        
        # If no tool calls, LLM has its final answer
        if not response.tool_calls:
            break
        
        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name in tool_registry:
                tool_result = tool_registry[tool_name].invoke(tool_args)
                tools_called.append({"tool": tool_name, "args": tool_args})
            else:
                tool_result = {"error": f"Tool {tool_name} not found"}
            
            # Add tool result to messages so LLM can see it
            messages.append(ToolMessage(
                content=json.dumps(tool_result, default=str),
                tool_call_id=tool_call["id"]
            ))
    
    state["draft_response"] = response.content
    state["tools_called"] = tools_called
    
    return state