//! nexus-core: Performance-critical components for Context Nexus

use pyo3::prelude::*;

mod token;
mod vector;
mod graph;
mod chunk;

#[pyfunction]
fn count_tokens(text: &str, _model: &str) -> PyResult<usize> {
    // rough estimate for now, proper tiktoken later
    Ok(text.len() / 4)
}

#[pyfunction]
fn score_vectors(query: Vec<f32>, documents: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    Ok(vector::cosine_batch(&query, &documents))
}

#[pyfunction]
#[pyo3(signature = (scores_a, scores_b, weight_a=0.5, weight_b=0.5, k=60))]
fn fuse_scores(scores_a: Vec<f32>, scores_b: Vec<f32>, weight_a: f32, weight_b: f32, k: usize) -> PyResult<Vec<f32>> {
    Ok(vector::rrf_fusion(&scores_a, &scores_b, weight_a, weight_b, k))
}

#[pyfunction]
#[pyo3(signature = (edges, start, max_depth=3, max_nodes=100))]
fn traverse_graph(edges: Vec<(usize, usize)>, start: usize, max_depth: usize, max_nodes: usize) -> PyResult<Vec<usize>> {
    Ok(graph::bfs(&edges, start, max_depth, max_nodes))
}

#[pyfunction]
#[pyo3(signature = (text, chunk_size=1000, overlap=100))]
fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> PyResult<Vec<String>> {
    Ok(chunk::split(text, chunk_size, overlap))
}

#[pymodule]
fn nexus_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(score_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(fuse_scores, m)?)?;
    m.add_function(wrap_pyfunction!(traverse_graph, m)?)?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    Ok(())
}
