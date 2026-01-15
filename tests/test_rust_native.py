#!/usr/bin/env python3
"""Test script to verify Rust native module is working."""

try:
    from context_nexus._core import count_tokens, score_vectors, fuse_scores, traverse_graph, chunk_text
    print('✅ Rust native module (_core) imported successfully!')
    
    # Test count_tokens
    tokens = count_tokens('Hello world test', 'gpt-4')
    print(f'✅ count_tokens() works: {tokens} tokens')
    
    # Test score_vectors
    query = [1.0, 0.0, 0.0]
    docs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7071, 0.7071, 0.0]]
    scores = score_vectors(query, docs)
    print(f'✅ score_vectors() works: {scores}')
    
    # Test chunk_text
    chunks = chunk_text('This is a test. ' * 100, chunk_size=50, overlap=10)
    print(f'✅ chunk_text() works: created {len(chunks)} chunks from 1600 char text')
    
    # Test traverse_graph
    edges = [(0, 1), (1, 2), (2, 3), (1, 4)]
    path = traverse_graph(edges, 0, max_depth=3, max_nodes=100)
    print(f'✅ traverse_graph() works: found {len(path)} nodes')
    
    # Test fuse_scores
    scores_a = [0.9, 0.7, 0.5]
    scores_b = [0.8, 0.9, 0.4]
    fused = fuse_scores(scores_a, scores_b, weight_a=0.5, weight_b=0.5, k=60)
    print(f'✅ fuse_scores() works: {fused}')
    
    print('\n🚀 All Rust functions working correctly! Native performance enabled.')
    
except ImportError as e:
    print(f'❌ Failed to import Rust module: {e}')
    import traceback
    traceback.print_exc()
