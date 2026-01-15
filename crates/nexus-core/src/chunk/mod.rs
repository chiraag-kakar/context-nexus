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
        let next_start = actual_end.saturating_sub(overlap);
        if next_start >= actual_end { break; }
        // Ensure start is on a char boundary
        start = adjust_to_char_boundary(text, next_start);
        if start >= actual_end { break; }
    }
    chunks
}

fn adjust_to_char_boundary(text: &str, pos: usize) -> usize {
    if pos >= text.len() {
        return text.len();
    }
    // Move forward to the next char boundary if we're in the middle of a char
    let mut adjusted = pos;
    while adjusted < text.len() && !text.is_char_boundary(adjusted) {
        adjusted += 1;
    }
    adjusted
}

fn find_break(text: &str, start: usize, mut end: usize) -> usize {
    if end >= text.len() { return text.len(); }
    
    // Ensure end is on a char boundary
    end = adjust_to_char_boundary(text, end);
    
    let chunk = &text[start..end];
    // try sentence boundary
    if let Some(p) = chunk.rfind(|c| c == '.' || c == '!' || c == '?') {
        let break_pos = start + p + 1;
        if p > chunk.len() / 2 && break_pos <= text.len() {
            return adjust_to_char_boundary(text, break_pos);
        }
    }
    // try word boundary
    if let Some(p) = chunk.rfind(' ') {
        let break_pos = start + p + 1;
        if break_pos <= text.len() {
            return adjust_to_char_boundary(text, break_pos);
        }
    }
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
