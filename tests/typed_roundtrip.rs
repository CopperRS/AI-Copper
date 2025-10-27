use ai_copper::FlowTensors;

#[test]
fn f64_roundtrip() {
    let vals: [f64; 4] = [1.0, -2.5, 3.25, 4.0];
    let dims = [2i64, 2i64];
    let t = FlowTensors::new_f64(&vals, &dims).expect("create f64 tensor");
    let out = t.data_f64().expect("read f64");
    assert_eq!(out, &vals);
}

#[test]
fn i32_roundtrip() {
    let vals: [i32; 4] = [1, -2, 30000, -12345];
    let dims = [2i64, 2i64];
    let t = FlowTensors::new_i32(&vals, &dims).expect("create i32 tensor");
    let out = t.data_i32().expect("read i32");
    assert_eq!(out, &vals);
}

#[test]
fn i64_roundtrip() {
    let vals: [i64; 3] = [1, -2, 9000000000];
    let dims = [3i64];
    let t = FlowTensors::new_i64(&vals, &dims).expect("create i64 tensor");
    let out = t.data_i64().expect("read i64");
    assert_eq!(out, &vals);
}
