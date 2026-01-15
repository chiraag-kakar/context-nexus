pub fn count(text: &str, _model: &str) -> Result<usize, String> {
    // TODO: use tiktoken-rs for proper counting
    Ok(text.len() / 4)
}

pub fn fits_budget(text: &str, budget: usize, model: &str) -> Result<bool, String> {
    Ok(count(text, model)? <= budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count() {
        assert!(count("hello world", "gpt-4").unwrap() > 0);
    }
}
