//! Benchmark for kernel functions

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn kernel_benchmarks(_c: &mut Criterion) {
    // TODO: Add benchmarks once kernels are implemented
}

criterion_group!(benches, kernel_benchmarks);
criterion_main!(benches);
