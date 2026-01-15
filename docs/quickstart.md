# Quickstart: Build Your First AI Agent in 15 Minutes

This guide takes you from zero to a working AI agent that can answer questions about your documents.

---

## Prerequisites

- **Python 3.10+** installed ([download](https://python.org/downloads))
- **OpenAI API key** ([get one free](https://platform.openai.com/api-keys))
- Some documents to ask questions about (PDFs, markdown, text files)

---

## Step 1: Set Up Your Project

Open your terminal and run:

```bash
# Create a new directory
mkdir my-ai-agent
cd my-ai-agent

# Create a virtual environment
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Context Nexus
pip install context-nexus
```

**Verify installation:**
```bash
python -c "import context_nexus; print('✅ Installed!')"
```

---

## Step 2: Add Your Documents

Create a folder for your documents:

```bash
mkdir docs
```

Add some files to the `docs/` folder. These can be:
- `.txt` files
- `.md` (Markdown) files
- `.pdf` files
- `.html` files

**Example:** Create a sample document:

```bash
cat > docs/company-policy.md << 'EOF'
# Company Refund Policy

## Standard Refunds
Customers can request a full refund within 30 days of purchase.
No questions asked.

## Extended Warranty Refunds
Products under extended warranty can be refunded within 90 days.
A receipt is required.

## Digital Products
Digital products are non-refundable after download.
Exceptions can be made for technical issues.
EOF
```

---

## Step 3: Set Your API Key

```bash
# On Mac/Linux:
export OPENAI_API_KEY="your-api-key-here"

# On Windows:
# set OPENAI_API_KEY=your-api-key-here
```

**Or create a `.env` file:**
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

---

## Step 4: Create Your Agent

Create a file called `agent.py`:

```python
"""My first AI agent with Context Nexus"""

import asyncio
from context_nexus import ContextNexus, Agent

async def main():
    # Step 1: Create the nexus
    print("🚀 Initializing Context Nexus...")
    nexus = ContextNexus()
    
    # Step 2: Ingest your documents
    print("📚 Ingesting documents...")
    await nexus.ingest(["./docs/"])
    print(f"   ✅ Indexed {nexus.stats.documents} documents")
    print(f"   ✅ Created {nexus.stats.chunks} chunks")
    
    # Step 3: Create an agent
    agent = Agent(nexus, token_budget=4000)
    
    # Step 4: Ask questions!
    print("\n🤖 Agent ready! Ask me anything about your documents.")
    print("   Type 'quit' to exit.\n")
    
    while True:
        query = input("You: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        answer = await agent.query(query)
        
        print(f"\nAgent: {answer.text}")
        print(f"\n📎 Sources:")
        for source in answer.sources[:3]:
            print(f"   - {source.title} ({source.relevance:.0%} relevant)")
        print()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Step 5: Run Your Agent

```bash
python agent.py
```

**Example conversation:**
```
🚀 Initializing Context Nexus...
📚 Ingesting documents...
   ✅ Indexed 1 documents
   ✅ Created 4 chunks

🤖 Agent ready! Ask me anything about your documents.
   Type 'quit' to exit.

You: What's the refund policy for digital products?

Agent: Digital products are non-refundable after download. However, 
exceptions can be made for technical issues.

📎 Sources:
   - company-policy.md (94% relevant)

You: How long do I have to request a refund?

Agent: For standard purchases, you have 30 days to request a full refund 
with no questions asked. For products under extended warranty, the refund 
window extends to 90 days, but a receipt is required.

📎 Sources:
   - company-policy.md (91% relevant)
```

---

## Step 6: Add Observability (Optional)

See what your agent is doing internally:

```python
# In your agent.py, modify the query call:
answer = await agent.query(query, trace=True)

print(f"\nAgent: {answer.text}")

# Show the trace
print(f"\n🔍 Trace:")
print(f"   Tokens used: {answer.trace.tokens_used}")
print(f"   Latency: {answer.trace.latency_ms}ms")
print(f"   Chunks retrieved: {len(answer.trace.chunks_retrieved)}")
```

---

## Step 7: Make It a Web App (Optional)

Create `app.py`:

```python
"""Simple web interface for your agent"""

from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from context_nexus import ContextNexus, Agent

app = FastAPI()

# Initialize on startup
nexus = None
agent = None

@app.on_event("startup")
async def startup():
    global nexus, agent
    nexus = ContextNexus()
    await nexus.ingest(["./docs/"])
    agent = Agent(nexus, token_budget=4000)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    answer = await agent.query(query.question)
    return {
        "answer": answer.text,
        "sources": [
            {"title": s.title, "relevance": s.relevance}
        ],
        "confidence": answer.confidence
    }

@app.get("/health")
def health():
    return {"status": "ok"}
```

**Run it:**
```bash
pip install fastapi uvicorn
uvicorn app:app --reload
```

**Test it:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

---

## Complete Project Structure

```
my-ai-agent/
├── venv/                 # Virtual environment
├── docs/                 # Your documents
│   └── company-policy.md
├── .env                  # API keys (don't commit!)
├── agent.py              # CLI agent
├── app.py                # Web API (optional)
└── requirements.txt      # Dependencies
```

**Create requirements.txt:**
```bash
echo "context-nexus" > requirements.txt
echo "fastapi" >> requirements.txt
echo "uvicorn" >> requirements.txt
```

---

## Next Steps

### Add More Documents
```python
await nexus.ingest([
    "./docs/",
    "./wiki/",
    "./policies/",
])
```

### Use Hybrid Retrieval
```python
from context_nexus import HybridRetriever

retriever = HybridRetriever(
    vector_weight=0.6,
    graph_weight=0.4,
)
agent = Agent(nexus, retriever=retriever)
```

### Get Structured Output
```python
from pydantic import BaseModel

class PolicyAnswer(BaseModel):
    applies_to: str
    refund_days: int
    conditions: list[str]

answer = await agent.query(
    "Summarize the refund policy",
    output_schema=PolicyAnswer
)
print(answer.structured)  # PolicyAnswer object
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'context_nexus'"
Make sure your virtual environment is activated:
```bash
source venv/bin/activate
```

### "Invalid API key"
Check your API key is set:
```bash
echo $OPENAI_API_KEY
```

### "No documents found"
Make sure your docs folder has files in it:
```bash
ls docs/
```

---

## Free Tier Limits

| Service | Free Tier | What You Can Do |
|---------|-----------|-----------------|
| **OpenAI** | $5 credit | ~500,000 tokens (~300 queries) |
| **Local Mode** | Unlimited | Use local embeddings + local LLM |

### Using 100% Free (No API Keys)

```python
nexus = ContextNexus(
    embeddings="local",    # Uses sentence-transformers
    llm="local",           # Uses Ollama or llama.cpp
)
```

Requires installing local models:
```bash
pip install context-nexus[local]
# Then download a model via Ollama
ollama pull llama3
```

---

## You Did It! 🎉

You now have a working AI agent that can answer questions about your documents.

**What you built:**
- ✅ Document ingestion pipeline
- ✅ Semantic search
- ✅ AI-powered Q&A
- ✅ Source citations
- ✅ (Optional) Web API

**Continue learning:**
- [Use Cases Guide](./use_cases.md) - Build more complex agents
- [Architecture](./architecture.md) - Understand how it works
- [Contributing](../CONTRIBUTING.md) - Help improve Context Nexus
