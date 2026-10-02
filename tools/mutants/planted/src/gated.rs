//! Code no build compiles while feature `x` is off: no `.d` file of the
//! build lists this file, so its mutants are UNBUILT, never MISSED.

/// Twice `n`.
pub fn twice(n: u32) -> u32 {
    n * 2
}

#[cfg(test)]
mod tests {
    #[test]
    fn twice() {
        assert_eq!(super::twice(3), 6);
    }
}
