use criterion::criterion_main;

mod multi;
mod ode;
mod single;

criterion_main!(single::benches, multi::benches, ode::benches);
