use llguidance::toktrie::SimpleVob;
use rand::Rng;

/// Hash bytes using FNV-1a to produce a 32-bit seed.
pub fn fnv1a_32(s: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c9dc5;
    for byte in s {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

/// Create a `SmallRng` seeded from a string via FNV-1a hash.
pub fn rng_from_str(s: &str) -> rand::rngs::SmallRng {
    use rand::SeedableRng;
    rand::rngs::SmallRng::seed_from_u64(fnv1a_32(s.as_bytes()) as u64)
}

/// Sample a random set-bit index from a `SimpleVob`.
///
/// Panics if the vob has no set bits.
pub fn sample_from_vob(rng: &mut impl Rng, vob: &SimpleVob) -> u32 {
    let nset = vob.num_set();
    assert!(nset > 0);
    if nset > vob.len() / 10 {
        loop {
            let idx = rng.random_range(0..vob.len());
            if vob[idx] {
                return idx as u32;
            }
        }
    } else {
        let choices = vob.to_list();
        choices[rng.random_range(0..choices.len())]
    }
}
