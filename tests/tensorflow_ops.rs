use ai_copper::FlowTensors;

#[test]
fn test_log1p_small() {
    let x = 1e-7_f32;
    let ft = FlowTensors::new(&[x], &[1]).expect("create");
    let out = ft.log1p().expect("log1p");
    let v = out.data().expect("data")[0];
    let expected = libm::log1pf(x);
    assert!((v - expected).abs() < 1e-8, "got {} expected {}", v, expected);
}

#[test]
fn test_sigmoid_zero() {
    let ft = FlowTensors::new(&[0.0_f32], &[1]).expect("create");
    let v = ft.sigmoid().expect("sig").data().unwrap()[0];
    assert!((v - 0.5).abs() < 1e-6, "sigmoid(0) = {}", v);
}

#[test]
fn test_comparisons() {
    let a = FlowTensors::new(&[0.0_f32, 1.0, 3.0], &[3]).unwrap();
    let b = FlowTensors::new(&[0.0_f32, 2.0, 2.0], &[3]).unwrap();

    let eq = a.equal(&b).unwrap();
    let eqs = eq.data().unwrap();
    assert_eq!(eqs[0], 1.0);
    assert_eq!(eqs[1], 0.0);
    assert_eq!(eqs[2], 0.0);

    let gt = a.greater(&b).unwrap();
    let gts = gt.data().unwrap();
    assert_eq!(gts[0], 0.0);
    assert_eq!(gts[1], 0.0);
    assert_eq!(gts[2], 1.0);

    let lt = a.less(&b).unwrap();
    let lts = lt.data().unwrap();
    assert_eq!(lts[0], 0.0);
    assert_eq!(lts[1], 1.0);
    assert_eq!(lts[2], 0.0);
}

#[test]
fn test_exp_ln_roundtrip() {
    let vals = [0.1_f32, 1.0, 2.5];
    let ft = FlowTensors::new(&vals, &[3]).unwrap();
    let out = ft.exp().unwrap().ln().unwrap();
    let got = out.data().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!((got[i] - v).abs() < 1e-6, "roundtrip {} vs {}", got[i], v);
    }
}
