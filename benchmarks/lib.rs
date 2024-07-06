use criterion::criterion_main;

mod single;
mod multi;

criterion_main!(
    single::benches, 
    multi::benches);
