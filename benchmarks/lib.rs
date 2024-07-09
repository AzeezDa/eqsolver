use criterion::criterion_main;

mod multi;
mod single;

criterion_main!(single::benches, multi::benches);
