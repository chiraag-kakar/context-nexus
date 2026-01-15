pub fn split(text: &str, size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || text.len() <= size {
        return if text.is_empty() { vec![] } else { vec![text.to_string()] };
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = (start + size).min(text.len());
        let actual_end = find_break(text, start, end);
        
        chunks.push(text[start..actual_end].to_string());
        
        if actual_end >= text.len() { break; }
        start = actual_end.saturating_sub(overlap);
        if start >= actual_end { break; }
    }
    chunks
}

fn find_break(text: &str, start: usize, end: usize) -> usize {
    if end >= text.len() { return text.len(); }
    
    let chunk = &text[start..end];
    // try sentence boundary
    if let Some(p) = chunk.rfind(|c| c == '.' || c == '!' || c == '?') {
        if p > chunk.len() / 2 { return start + p + 1; }
    }
    // try word boundary
    if let Some(p) = chunk.rfind(' ') { return start + p + 1; }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_short() {
        assert_eq!(split("hi", 100, 10).len(), 1);
    }
}
