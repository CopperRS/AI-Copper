use std::env;
use ai_copper::FlowTensors;

#[test]
fn string_roundtrip_placeholder() {
    // Quick test: create a small string tensor and read back
    let vals = ["hello", "from", "rust"];
    let dims = [3i64];
    let t = FlowTensors::new_string(&vals, &dims).expect("create string tensor");
    let out = t.data_strings().expect("read strings");
    assert_eq!(out, vec!["hello".to_string(), "from".to_string(), "rust".to_string()]);
}
