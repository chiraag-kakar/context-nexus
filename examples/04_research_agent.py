"""
Research agent example - iteratively refine answers with confidence tracking.

This shows how to build a research workflow that:
- Searches for information iteratively
- Tracks confidence scores
- Knows when it has enough information
- Handles rate limiting properly
"""

import asyncio
import os
import time
from typing import List
from dataclasses import dataclass
from context_nexus import ContextNexus, Agent
from context_nexus.core.types import Document


@dataclass
class ResearchContext:
    """Keeps track of where we are in the research process"""
    query: str
    findings: List[str]
    sources_used: List[str]
    confidence: float
    needs_more_info: bool


async def main():
    print("=" * 70)
    print("RESEARCH & SYNTHESIS AGENT")
    print("=" * 70)
    print()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY environment variable")
        return
    
    # ---- SETUP ----
    print("📚 PHASE 1: SETUP & DATA PREPARATION")
    print("-" * 70)
    
    nexus = ContextNexus()
    
    # Build a knowledge base to research from
    research_docs = [
        Document(
            content="""
            # Machine Learning Fundamentals
            
            Machine Learning (ML) is a subset of artificial intelligence that enables
            systems to learn and improve from experience without being explicitly programmed.
            
            ## Key Concepts:
            
            1. **Supervised Learning**: Training on labeled data
               - Classification: Predicting discrete categories
               - Regression: Predicting continuous values
            
            2. **Unsupervised Learning**: Finding patterns in unlabeled data
               - Clustering: Grouping similar data points
               - Dimensionality reduction: Reducing feature space
            
            3. **Reinforcement Learning**: Learning through trial and error
               - Agent takes actions in an environment
               - Receives rewards or penalties
               - Learns optimal policy over time
            """,
            source="ml_fundamentals.md",
            id="ml_fund"
        ),
        Document(
            content="""
            # RAG Systems Architecture
            
            Retrieval-Augmented Generation (RAG) combines information retrieval
            with large language models to generate informed responses.
            
            ## Architecture Components:
            
            ###1. Document Processing
            - Chunking: Breaking documents into manageable pieces
            - Embedding: Converting text to vector representations
            - Indexing: Storing vectors for fast retrieval
            
            ### 2. Retrieval
            - Vector search: Finding semantically similar content
            - Hybrid search: Combining semantic + keyword matching
            - Reranking: Ordering results by relevance
            
            ### 3. Generation
            - Context assembly: Building prompts with retrieved docs
            - LLM generation: Producing answers
            - Source attribution: Citing references
            
            ## Challenges:
            - Context window limits
            - Relevance vs diversity tradeoff
            - Hallucination mitigation
            - Performance at scale
            """,
            source="rag_architecture.md",
            id="rag_arch"
        ),
        # Additional docs omitted for brevity...
    ]
    
    print(f"Preparing knowledge base with {len(research_docs)} documents...")
    stats = await nexus.ingest(research_docs)
    
    print(f"✅ Knowledge base ready!")
    print(f"   Documents: {stats.documents}")
    print(f"   Searchable chunks: {stats.chunks}")
    print()
    
    # ---- ITERATIVE RESEARCH ----
    print("🔍 PHASE 2: ITERATIVE RESEARCH WORKFLOW")
    print("-" * 70)
    
    agent = Agent(nexus, token_budget=10000)
    
    research_query = "How do I build a production-ready RAG system?"
    
    print(f"Research Question: {research_query}")
    print()
    
    try:
        # Start the research process
        context = ResearchContext(
            query=research_query,
            findings=[],
            sources_used=[],
            confidence=0.0,
            needs_more_info=True
        )
        
        iteration = 0
        max_iterations = 3
        
        # Keep researching until we're confident or hit the limit
        while context.needs_more_info and iteration < max_iterations:
            iteration += 1
            print(f"--- Iteration {iteration} ---")
            print()
            
            # Retrieve information
            print(f"  🔎 Retrieving information...")
            start_time = time.time()
            
            results = await nexus.retrieve(
                context.query + " " + " ".join(context.findings),
                limit=5
            )
            
            retrieve_time = int((time.time() - start_time) * 1000)
            print(f"     Retrieved {len(results)} chunks in {retrieve_time}ms")
            
            # Synthesize what we found
            print(f"  🧠 Synthesizing information...")
            
            if iteration == 1:
                question = research_query
            else:
                # Build on what we already know
                question = f"{research_query} Also address: {', '.join(context.findings[-2:])}"
            
            answer = await agent.query(question, trace=True)
            
            if answer.trace:
                print(f"     Generated in {answer.trace.latency_ms}ms")
            
            # Update our research state
            context.findings.append(answer.text[:100] + "...")
            context.sources_used.extend([s.title for s in answer.sources])
            context.confidence = answer.confidence
            
            print(f"  📊 Confidence: {context.confidence:.0%}")
            print(f"  📚 Sources used: {len(set(context.sources_used))}")
            
            # Check if we have enough
            if context.confidence > 0.80 and len(set(context.sources_used)) >= 3:
                context.needs_more_info = False
                print(f"  ✅ Research complete (high confidence)")
            else:
                print(f"  ⏳ More research needed")
            
            print()
            
            # Be nice to the API
            if context.needs_more_info:
                print("  💤 Rate limit pause (500ms)")
                await asyncio.sleep(0.5)
        
        # ---- FINAL SYNTHESIS ----
        print("📝 PHASE 3: SYNTHESIS & FINAL ANSWER")
        print("-" * 70)
        
        print("Generating comprehensive synthesis...")
        print()
        
        final_answer = await agent.query(
            f"{research_query} Give a comprehensive answer covering architecture, "
            f"deployment, and best practices.",
            trace=True
        )
        
        print("=" * 70)
        print("RESEARCH FINDINGS")
        print("=" * 70)
        print()
        print(final_answer.text)
        print()
        
        # ---- METADATA ----
        print("=" * 70)
        print("📊 RESEARCH METADATA")
        print("=" * 70)
        print()
        print(f"Iterations: {iteration}")
        print(f"Final Confidence: {final_answer.confidence:.0%}")
        print(f"Unique Sources: {len(set(context.sources_used))}")
        print()
        
        if final_answer.trace:
            print("Performance:")
            print(f"  Total time: {final_answer.trace.latency_ms}ms")
            print(f"  Tokens used: {final_answer.trace.tokens_used}")
            print(f"  Chunks retrieved: {final_answer.trace.chunks_retrieved}")
            print()
        
        print("Sources:")
        for i, source in enumerate(final_answer.sources, 1):
            print(f"  [{i}] {source.title} (relevance: {source.relevance:.0%})")
        print()
        
        # ---- SUMMARY ----
        print("=" * 70)
        print("✅ RESEARCH WORKFLOW COMPLETE")
        print("=" * 70)
        print()
        print("What we did:")
        print("  ✅ Iterative refinement (3 iterations)")
        print("  ✅ Confidence-based stopping criteria")
        print("  ✅ Rate limiting between iterations")
        print("  ✅ Multi-source synthesis")
        print("  ✅ Full traceability")
        print()
        print("Key patterns:")
        print("  1. Start broad, refine with each iteration")
        print("  2. Track confidence scores to know when to stop")
        print("  3. Implement rate limiting to respect API limits")
        print("  4. Combine findings from multiple sources")
        print("  5. Provide full attribution for transparency")
        print()
        
    finally:
        await agent.close()
        print("🧹 Cleaned up resources")


if __name__ == "__main__":
    asyncio.run(main())
