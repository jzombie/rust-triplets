use std::error::Error;

#[path = "common/example_sources.rs"]
mod example_sources;

fn main() -> Result<(), Box<dyn Error>> {
    triplets::example_apps::run_multi_source_demo(
        std::env::args().skip(1),
        example_sources::resolve_source_roots,
        example_sources::build_default_sources,
    )
}
