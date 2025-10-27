use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ai_copper::FlowTensors;

fn bench_map_seq_vs_par(c: &mut Criterion) {
    let n = 1 << 20; // 1M elements
    let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0001).collect();
    let ft = FlowTensors::new(&vals, &[n as i64]).expect("create");

    c.bench_function("map_seq_exp", |b| {
        b.iter(|| {
            let out = ft.map(|x| x.exp()).expect("map");
            black_box(out);
        })
    });

    c.bench_function("map_par_exp", |b| {
        b.iter(|| {
            // same call uses rayon internally when large
            let out = ft.map(|x| x.exp()).expect("map");
            black_box(out);
        })
    });
}

criterion_group!(benches, bench_map_seq_vs_par);
criterion_main!(benches);
