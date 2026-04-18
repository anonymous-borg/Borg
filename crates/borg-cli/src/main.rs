use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use borg_core::types::Request;
use borg_core::{Engine, Graph, System};
use serde::Deserialize;

mod write_jsonl;

use crate::write_jsonl::write_results_jsonl;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    workload: PathBuf,
    #[arg(long)]
    output_jsonl: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    system: Box<dyn System>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config_path = args.config;
    let workload_path = args.workload;
    let mut config = load_config(&config_path)?;
    let system_base_path = config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    config.system.resolve_path(&system_base_path)?;
    config.system.init()?;
    let mut graph = Graph::new();
    load_workload(config.system.as_mut(), &mut graph, &workload_path)?;

    let engine = Engine::new(graph, config.system);
    let results = engine.run()?;

    if let Some(parent) = args.output_jsonl.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory {}", parent.display())
            })?;
        }
    }
    write_results_jsonl(&args.output_jsonl, &results)?;

    Ok(())
}

fn load_config(config_path: &Path) -> Result<Config> {
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    toml::from_str(&raw).with_context(|| format!("failed to parse {}", config_path.display()))
}

fn load_workload(system: &mut dyn System, graph: &mut Graph, workload_path: &Path) -> Result<()> {
    let file = std::fs::File::open(workload_path)
        .with_context(|| format!("failed to open {}", workload_path.display()))?;
    let reader = std::io::BufReader::new(file);

    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from {}",
                line_index + 1,
                workload_path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }

        let request: Request = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse workload row {} from {}",
                line_index + 1,
                workload_path.display()
            )
        })?;
        request.validate().with_context(|| {
            format!(
                "invalid workload row {} from {}",
                line_index + 1,
                workload_path.display()
            )
        })?;
        system.add_request_arrival(request, graph, None, None)?;
    }
    Ok(())
}
