use ai_copper::FlowTensors;

// Helper to build interleaved complex vector from pairs
fn interleaved(pairs: &[(f32, f32)]) -> Vec<f32> {
    let mut v = Vec::with_capacity(pairs.len() * 2);
    for (r, i) in pairs { v.push(*r); v.push(*i); }
    v
}

#[test]
fn complex64_add_mul() {
    // a = [1+2i, 3+4i]
    let a_pairs = [(1.0f32, 2.0f32), (3.0, 4.0)];
    let b_pairs = [(5.0f32, 6.0f32), (7.0, 8.0)];
    let a_data = interleaved(&a_pairs);
    let b_data = interleaved(&b_pairs);
    let dims = [2i64];
    let ta = FlowTensors::new_complex64(&a_data, &dims).expect("create a");
    let tb = FlowTensors::new_complex64(&b_data, &dims).expect("create b");

    // add
    let res_add = ta.add(&tb).expect("add");
    let out_add = res_add.data_complex64().expect("read complex");
    // expected (1+5, 2+6, 3+7, 4+8)
    let expected_add = interleaved(&[(6.0, 8.0), (10.0, 12.0)]);
    assert_eq!(out_add, &expected_add[..]);

    // mul
    let res_mul = ta.mul(&tb).expect("mul");
    let out_mul = res_mul.data_complex64().expect("read complex mul");
    // multiply first elements: (1+2i)*(5+6i) = 1*5 - 2*6 + i(1*6 + 2*5) = (5-12, 6+10) = (-7,16)
    let expected_mul = interleaved(&[(-7.0, 16.0), (-11.0, 52.0)]); // second: (3+4i)*(7+8i) = (21-32, 24+28) = (-11,52)
    assert_eq!(out_mul, &expected_mul[..]);
}
