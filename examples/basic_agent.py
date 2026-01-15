"""Basic example - document Q&A"""

import asyncio
from pathlib import Path
from context_nexus import ContextNexus, Agent


async def main():
    print("Context Nexus - Basic Example\n")

    # create sample doc
    sample_dir = Path("./sample_docs")
    sample_dir.mkdir(exist_ok=True)
    (sample_dir / "policy.md").write_text("""
# Company Policy

## Refunds
- Full refund within 30 days
- Extended warranty: 90 days with receipt

## Shipping
- Free over $50
- Standard: 5-7 days
- Express: 1-2 days
""")

    nexus = ContextNexus()
    await nexus.ingest([sample_dir])
    print(f"Indexed {nexus.stats.documents} docs, {nexus.stats.chunks} chunks\n")

    agent = Agent(nexus)

    for q in ["What's the refund policy?", "How long does shipping take?"]:
        print(f"Q: {q}")
        ans = await agent.query(q)
        print(f"A: {ans.text}\n")

    # cleanup
    (sample_dir / "policy.md").unlink()
    sample_dir.rmdir()


if __name__ == "__main__":
    asyncio.run(main())
