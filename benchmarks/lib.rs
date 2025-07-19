use criterion::criterion_main;

mod multi;
mod ode;
mod single;
mod integrators;

criterion_main!(single::benches, multi::benches, ode::benches, integrators::benches);
