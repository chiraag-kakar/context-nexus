use std::collections::{HashSet, VecDeque, HashMap};

pub fn bfs(edges: &[(usize, usize)], start: usize, max_depth: usize, max_nodes: usize) -> Vec<usize> {
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(a, b) in edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }

    let mut seen = HashSet::new();
    let mut result = Vec::new();
    let mut q = VecDeque::new();
    
    q.push_back((start, 0));
    seen.insert(start);

    while let Some((node, depth)) = q.pop_front() {
        if result.len() >= max_nodes { break; }
        result.push(node);
        
        if depth >= max_depth { continue; }
        
        for &neighbor in adj.get(&node).unwrap_or(&vec![]) {
            if seen.insert(neighbor) {
                q.push_back((neighbor, depth + 1));
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs() {
        let edges = vec![(0, 1), (1, 2)];
        let r = bfs(&edges, 0, 2, 10);
        assert!(r.contains(&0) && r.contains(&1));
    }
}
