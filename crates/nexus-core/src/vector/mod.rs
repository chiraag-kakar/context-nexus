use rayon::prelude::*;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() { return 0.0; }
    
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

pub fn cosine_batch(query: &[f32], docs: &[Vec<f32>]) -> Vec<f32> {
    docs.par_iter().map(|d| cosine(query, d)).collect()
}

fn ranks(scores: &[f32]) -> Vec<usize> {
    let mut indexed: Vec<_> = scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0; scores.len()];
    for (rank, (idx, _)) in indexed.into_iter().enumerate() {
        r[idx] = rank + 1;
    }
    r
}

pub fn rrf_fusion(a: &[f32], b: &[f32], wa: f32, wb: f32, k: usize) -> Vec<f32> {
    let ra = ranks(a);
    let rb = ranks(b);
    let kf = k as f32;
    ra.iter().zip(&rb).map(|(&x, &y)| wa / (kf + x as f32) + wb / (kf + y as f32)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 0.01);
    }
}
