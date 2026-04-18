use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Result;

use borg_core::types::RequestResult;

pub fn write_results_jsonl(path: &Path, results: &[RequestResult]) -> Result<()> {
    let mut rows = results.to_vec();
    rows.sort_by_key(|row| row.request_id);

    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    for row in &rows {
        serde_json::to_writer(&mut writer, row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}
