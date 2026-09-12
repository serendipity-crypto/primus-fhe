//! Maps odd-root evaluations through a backend's Fourier storage order.

/// Both bundled backends evaluate at exp(i*pi*(1-4*j)/N), j=0..N/2.
/// An odd exponent maps these roots either to stored roots or their conjugates.
/// The caller supplies a supported power-of-two `n` and a bijection from
/// standard frequency indices to storage indices in `0..n/2`.
pub(crate) fn automorphism_map(
    n: usize,
    degree: usize,
    storage_index: impl Fn(usize) -> usize,
) -> Vec<(usize, bool)> {
    let two_n = n
        .checked_mul(2)
        .expect("FFT automorphism ring length overflow");
    assert!(
        degree < two_n && degree % 2 == 1,
        "FFT automorphism degree must be odd and less than 2N"
    );
    let mask = two_n - 1;
    let mut map = vec![(0, false); n / 2];
    for j in 0..n / 2 {
        // Wrapping arithmetic is exact modulo the power-of-two 2N.
        let root = 1usize.wrapping_sub(4usize.wrapping_mul(j));
        let mut mapped = degree.wrapping_mul(root) & mask;
        let conjugate = mapped % 4 == 3;
        if conjugate {
            mapped = mapped.wrapping_neg() & mask;
        }
        let source = (1usize.wrapping_sub(mapped) & mask) / 4;
        map[storage_index(j)] = (storage_index(source), conjugate);
    }
    map
}
