# Assistly : Multi-Agent AI Customer Support System

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge&logo=streamlit)](https://assistly-app.streamlit.app/)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Assistly** is a multi-agent AI customer support system designed for modern e-commerce platforms. Unlike traditional chat deflections that merely regurgitate policy pages, Assistly is built with backend integration. It acts as an autonomous support engineer: lookup up user records, updating order statuses, triggering refunds, dispatching replacements, managing inventory levels, and routing tickets to human operators when things go wrong.

---

## 📌 Table of Contents

- [😤 Why Assistly?](#-why-assistly)
- [🗺️ Architecture Flow](#️-architecture-flow)
- [🤖 The Multi-Agent Squad](#-the-multi-agent-squad)
  - [1. 🔍 Classifier Agent](#1--classifier-agent)
  - [2. 🛠️ Resolver Agent (ReAct Tool-Calling)](#2--resolver-agent-react-tool-calling)
  - [3. ✅ QA Agent (Auditor)](#3--qa-agent-auditor)
  - [4. 🚨 Escalation Agent (Rules Engine)](#4--escalation-agent-rules-engine)
- [🧠 Fine-tuned Local Classifier](#-fine-tuned-local-classifier)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 How It Works: A Step-by-Step Example](#-how-it-works-a-step-by-step-example)
- [⚙️ Getting Started](#️-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [🗂️ Project Structure](#️-project-structure)
- [🌍 Live Demo](#-live-demo)
- [✍️ Creator & Vision](#️-creator--vision)

---


## 😤 Why Assistly?

Most customer support chatbots are incredibly frustrating. You’ve probably interacted with them: you ask for a refund, the bot correctly identifies that you are unhappy, and then it simply prints a link to a return policy page or tells you to call a support hotline. According to HubSpot, **75% of customers expect help within 5 minutes**, yet traditional chatbots average a poor **~30% resolution rate** because they cannot execute real backend actions. 

When you consider that the average cost of a human support agent is **$25 to $40 per hour**, handling thousands of tickets daily becomes a massive cost center. Deflecting them with bad bots isn’t the answer either; Qualtrics reports that customers who experience repeated poor support interactions are **3x more likely to churn**. 

To make matters worse, typical AI systems treat every customer the exact same way. A first-time buyer with a simple question gets the same standard treatment as a gold-tier member who has faced three delays in a row, even though the latter is on the verge of taking their business elsewhere. 

**I built Assistly to bridge this gap.** It uses a state-driven multi-agent architecture to classify support requests, safely perform transactional operations on backend databases, check returns policies, dynamically check customer frustration history, and offer specialized flows for VIPs.

---

## 🗺️ Architecture Flow

Assistly coordinates multiple agents using a LangGraph state machine. Below is the workflow diagram illustrating how a user request traverses the system:

```
                               +-------------------+
                               |     Customer      |
                               +---------+---------+
                                         |
                                         v
                               +---------+---------+
                               |   Streamlit UI    |
                               +---------+---------+
                                         |
                                         v
                               +---------+---------+
                               |   Orchestrator    |
                               +---------+---------+
                                         | (Shared TypedDict State)
     +-----------------------------------+-----------------------------------+
     |                                   |                                   |
     v                                   v                                   v
+----+------+                      +-----+-----+                       +-----+-----+
| Classifier|                      |  Resolver |                       | QA Agent  |
| (Qwen PEFT|                      |  (ReAct   |                       | (Auditor  |
| / Groq)   |                      |  Engine)  |                       |  Scorer)  |
+-----------+                      +-----+-----+                       +-----------+
                                         |                                   |
                                         v                                   |
                                   +-----+-----+                             |
                                   |Escalation |<----------------------------+
                                   | (Rules)   |
                                   +-----+-----+
                                         |
                                         v
                               +---------+---------+
                               |  Final Response   |
                               +-------------------+
```

---

## 🤖 The Multi-Agent Squad

Assistly uses four specialized agents, each focused on a single responsibility:

### 1. 🔍 Classifier Agent
* **Objective:** Understand what the customer wants, how they feel, and what records they are talking about.
* **Extraction Capabilities:** Detects intent across 9 categories (e.g., `order_status`, `refund_request`, `shipping_inquiry`, `product_complaint`, `cancellation_request`, `human_request`), sentiment (`positive`, `neutral`, `negative`, `angry`), urgency, and a numerical frustration score (0.0 to 1.0).
* **Robust Safety Nets:** Features a deterministic regex-based order ID extraction parser to prevent the LLM from hallucinating order numbers.
* **Hybrid Execution:** Primary inference is offloaded to a local fine-tuned Qwen2.5-1.5B model. If local model weights are missing, it falls back to the Groq Cloud API.

### 2. 🛠️ Resolver Agent (ReAct Tool-Calling)
* **Objective:** Solve the customer's issue by querying data or taking transactional action.
* **Tool suite:** Interacts with the backend via 9 tools: `get_order_by_id`, `get_orders_by_customer_email`, `get_order_journey_details`, `get_customer_incident_profile`, `process_refund`, `get_refund_status`, `process_replacement`, `get_replacement_status`, and `search_knowledge_base`.
* **Guardrails & Policies:** Bypasses or restricts actions based on corporate policies:
  * **Electronics policy:** If an item is `replacement_only`, the resolver blocks cash refunds and directs the user to a priority replacement.
  * **Hygiene/Clearance policy:** If an item is `non_returnable`, it blocks returns and offers a ₹500 store credit coupon instead.
  * **Frictionless VIP Routing:** If `get_customer_incident_profile` reveals 2+ past incidents, standard confirmation loops are bypassed, and the customer is instantly refunded or replaced.
  * **Multi-turn Confirmation:** For standard customers, it halts before destructive operations, prompts for the cancellation reason, saves it, and asks for confirmation.

### 3. ✅ QA Agent (Auditor)
* **Objective:** Ensure the response draft is accurate, empathetic, and factual.
* **Factual Check:** cross-references the LLM draft against the SQL database ground truth (e.g., checking if the refund was actually committed).
* **Score-Based Rewriting:** Scores drafts on a scale from 0.0 to 1.0. If the score is $\ge 0.75$, the response passes through. If not, the QA agent rewrites the response to fix details.

### 4. 🚨 Escalation Agent (Rules Engine)
* **Objective:** Decide if a human operator needs to take over.
* **Rules:**
  * **Rule 1:** Never escalate if the user is in the middle of a confirmation flow.
  * **Rule 2:** Always escalate on explicit `human_request` intent.
  * **Rule 3:** Never auto-escalate basic, successful refunds.
  * **Rule 4:** Escalate if customer frustration is high and the QA score remains low.
  * **Rule 5:** Escalate immediately on high-urgency messages with angry sentiment.

---

## 🧠 Fine-tuned Local Classifier

To avoid cloud API latency and ensure data privacy, the Classifier runs on a locally fine-tuned language model:
* **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
* **Training Method:** QLoRA (4-bit quantization, rank $r=16$, $\alpha=32$, targeting all linear modules) to enable running on consumer-grade GPUs.
* **Dataset:** Trained on the *Bitext Customer Support Dataset* (27,000 rows across 27 intents, mapped down to 9 primary labels).
* **Execution:** Run locally via HuggingFace Transformers and PEFT. If weights are not found under `fine_tuned_classifier/`, the system automatically falls back to Groq's cloud-hosted models.

---

## 🛠️ Tech Stack

* **Orchestration:** ![LangGraph](https://img.shields.io/badge/LangGraph-black?style=flat-square) ![LangChain](https://img.shields.io/badge/LangChain-blue?style=flat-square)
* **LLM Engine:** ![Groq API](https://img.shields.io/badge/Groq_API-black?style=flat-square) ![Transformers](https://img.shields.io/badge/Transformers-orange?style=flat-square) ![PEFT](https://img.shields.io/badge/PEFT-orange?style=flat-square) ![Qwen](https://img.shields.io/badge/Qwen--1.5B-green?style=flat-square)
* **Database:** ![SQLite](https://img.shields.io/badge/SQLite-blue?style=flat-square) ![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-red?style=flat-square) ![ChromaDB](https://img.shields.io/badge/ChromaDB-yellow?style=flat-square)
* **Frontend:** ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square)

---

## 🚀 How It Works: A Step-by-Step Example

Let's walk through what happens when a standard customer sends: 
`"My order ORD00188 hasn't arrived. I want a refund."`

```
[Customer message] 
  |
  +---> [Classifier Agent]
  |       * Identifies intent: "cancellation_request"
  |       * Extracts order_id: "ORD00188"
  |       * Detects sentiment: "negative"
  |
  +---> [Resolver Agent]
  |       * Queries `get_order_by_id` -> Status is "processing"
  |       * Queries `get_customer_incident_profile` -> Incident count is 0 (Standard customer)
  |       * Evaluates return policy -> Item is eligible
  |       * Intercepts tool execution -> Brakes to ask for a cancellation reason:
  |         "I can help you cancel ORD00188. Please choose a reason: Delayed Delivery, 
  |          Damaged Product, Ordered by Mistake, or Found Better Price."
  |
[User replies: "It is a Delayed Delivery"]
  |
  +---> [Resolver Agent (Turn 2)]
  |       * Maps reason to "Delayed Delivery"
  |       * Holds execution to ask for confirmation:
  |         "I can process a full refund. Would you like me to proceed? (Yes/No)"
  |
[User replies: "Yes, proceed"]
  |
  +---> [Resolver Agent (Turn 3)]
          * Invokes `process_refund(order_id='ORD00188', reason='Delayed Delivery')`
          * Writes to DB (engine.begin() atomic block):
              1. Adds refund record to `refunds` with reason "Delayed Delivery".
              2. Updates status in `orders` to "cancelled".
              3. Increments product stock in `inventory` by +1.
              4. Logs status change and audit details to `order_events`.
          * Returns success response draft.
```

---

## ⚙️ Getting Started

### Prerequisites
* Python 3.10 or 3.11
* CUDA-compatible GPU (Optional, required only for local PEFT classifier execution)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sanjana006/Assistly-AI-Customer-Support-System
   cd Assistly-AI-Customer-Support-System
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   # DATABASE_URL=sqlite:///./multi-agent-support/data/support.db
   # CHROMA_PATH=./multi-agent-support/knowledge_base/chroma_db
   ```

4. **Initialize and Seed the Database:**
   This sets up the SQLite schema, seeds 100 customers, 500 mock orders, and populates the inventory tables:
   ```bash
   python multi-agent-support/data/seed_database.py
   ```

5. **(Optional) Run Fine-Tuning Pipeline:**
   If you want to train and load the local QLoRA classifier:
   ```bash
   python multi-agent-support/scripts/load_dataset.py
   python multi-agent-support/scripts/finetune.py
   ```

6. **Start the Web UI:**
   ```bash
   streamlit run multi-agent-support/ui/app.py
   ```

---

## 🗂️ Project Structure

```
Assistly-AI-Customer-Support-System/
├── multi-agent-support/
│   ├── agents/
│   │   ├── classifier.py       # Intent + sentiment detection
│   │   ├── resolver.py         # Tool-calling ReAct agent
│   │   ├── qa_agent.py         # Response quality scorer
│   │   ├── escalation.py       # Rules-based escalation engine
│   │   └── orchestrator.py     # LangGraph state machine
│   ├── tools/
│   │   ├── order_lookup.py     # Order + customer DB tools
│   │   ├── refund_processor.py # Refund + replacement + audit tools
│   │   └── kb_search.py        # ChromaDB vector search
│   ├── data/
│   │   ├── seed_database.py    # Seeds 100 customers, 500 orders, and inventory
│   │   └── load_dataset.py     # Downloads Bitext dataset
│   ├── scripts/
│   │   └── finetune.py         # QLoRA fine-tuning pipeline
│   ├── tests/                  # Unit tests for all agents
│   ├── fine_tuned_classifier/  # Saved LoRA adapter weights
│   └── app.py                  # Streamlit UI
├── config.py
├── .env.example
└── requirements.txt
```

---

## 🌍 Live Demo

The application is deployed live on Streamlit Community Cloud. You can try out standard refund flows, VIP bypass scenarios, and policy questions in real-time.

### [🚀 Try Assistly Live →](https://assistly-app.streamlit.app/)

---

## ✍️ Creator & Vision

**Sanjana Nathani**  
*AI Engineer & Systems Developer*

> "Conversational AI shouldn't just deflect; it should resolve."

Assistly was born out of a desire to push conversational AI beyond simple rule-based deflections. For too long, customer support chatbots have been isolated from transactional backend systems—relegated to quoting static policy pages rather than resolving real issues. 

By designing Assistly, I wanted to demonstrate how multi-agent orchestrations (using **LangGraph**), structured tool-calling loops (**ReAct**), and fine-tuned local LLMs (**Qwen-1.5B via QLoRA**) can converge to create a safe, transaction-aware, and context-sensitive customer support engine that delivers genuine business value.

### 🤝 Connect & Collaborate
* 🖥️ **GitHub**: [@Sanjana006](https://github.com/Sanjana006)
* 💼 **LinkedIn**: [Sanjana Nathani](https://www.linkedin.com/in/sanjana-nathani-26a42727b/)
* 📧 **Get in Touch**: Feel free to open an issue or submit a pull request if you want to collaborate!
