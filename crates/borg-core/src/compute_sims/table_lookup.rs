use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use csv::ReaderBuilder;
use serde::Deserialize;

use crate::compute_sim::ComputeSimulator;
use crate::module::Module;
use crate::types::{ComputeContext, Layer, ParallelStrategy, RequestState};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TableCsvRow {
    name: String,
    tp: u32,
    key_0: u64,
    key_1: u64,
    latency: f64,
}

#[derive(Debug, Default, Clone, Copy)]
struct LatencyAccumulator {
    total: f64,
    count: u32,
}

type Key1Samples = BTreeMap<u64, LatencyAccumulator>;
type Key0Samples = BTreeMap<u64, Key1Samples>;
type LayerSamples = HashMap<String, Key0Samples>;
type TpLayerSamples = HashMap<u32, LayerSamples>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LookupKey {
    key_0: u64,
    key_1: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct BatchKeys {
    tokens: u64,
    lm_heads: u64,
    attention_compute: u64,
    attention_memory: u64,
}

impl BatchKeys {
    fn from_batch(batch: &[RequestState]) -> Self {
        let mut keys = Self::default();
        for request in batch {
            let q_len = u64::from(request.q_len);
            let kv_len = u64::from(request.kv_len);
            keys.tokens += q_len;
            keys.lm_heads += u64::from(request.lm_head_len);
            keys.attention_memory += kv_len;
            keys.attention_compute += q_len * (q_len / 2 + kv_len);
        }
        keys
    }
}

impl LatencyAccumulator {
    fn record(&mut self, latency: f64) {
        self.total += latency;
        self.count += 1;
    }

    fn mean(self) -> Result<f64> {
        if self.count == 0 {
            bail!("latency accumulator is empty");
        }
        Ok(self.total / f64::from(self.count))
    }
}

#[derive(Debug, Clone)]
struct LookupTable1D {
    keys: Vec<u64>,
    values: Vec<f64>,
}

impl LookupTable1D {
    fn from_samples(samples: BTreeMap<u64, f64>, label: &str) -> Result<Self> {
        if samples.is_empty() {
            bail!("{label} table is empty");
        }

        let (keys, values): (Vec<_>, Vec<_>) = samples.into_iter().unzip();
        Ok(Self { keys, values })
    }

    fn lookup(&self, query: u64) -> f64 {
        let (lower, upper) = lookup_bounds(&self.keys, query);
        let lower_value = self.values[lower];
        if lower == upper {
            return lower_value;
        }

        linear_interpolate(
            self.keys[lower],
            lower_value,
            self.keys[upper],
            self.values[upper],
            query,
        )
    }
}

#[derive(Debug, Clone)]
struct LookupTable2D {
    key_0: Vec<u64>,
    rows: Vec<LookupTable1D>,
}

impl LookupTable2D {
    fn from_samples(samples: BTreeMap<u64, BTreeMap<u64, f64>>, label: &str) -> Result<Self> {
        if samples.is_empty() {
            bail!("{label} table is empty");
        }

        let mut key_0 = Vec::with_capacity(samples.len());
        let mut rows = Vec::with_capacity(samples.len());
        for (outer_key, inner_samples) in samples {
            key_0.push(outer_key);
            rows.push(LookupTable1D::from_samples(
                inner_samples,
                &format!("{label} inner row {outer_key}"),
            )?);
        }

        Ok(Self { key_0, rows })
    }

    fn lookup(&self, key_0_query: u64, key_1_query: u64) -> f64 {
        let (lower, upper) = lookup_bounds(&self.key_0, key_0_query);
        let lower_value = self.rows[lower].lookup(key_1_query);
        if lower == upper {
            return lower_value;
        }

        linear_interpolate(
            self.key_0[lower],
            lower_value,
            self.key_0[upper],
            self.rows[upper].lookup(key_1_query),
            key_0_query,
        )
    }
}

#[derive(Debug)]
struct TpLookupTable {
    layer_latencies_s: HashMap<String, LookupTable2D>,
}

impl TpLookupTable {
    fn lookup(&self, layer_name: &str, key: LookupKey) -> Result<f64> {
        let samples = self
            .layer_latencies_s
            .get(layer_name)
            .ok_or_else(|| anyhow!("unknown layer '{layer_name}'"))?;
        let latency = samples.lookup(key.key_0, key.key_1);
        validate_latency(latency, layer_name)?;
        Ok(latency)
    }
}

#[derive(Debug)]
struct TraceLookupTable {
    tp_tables: HashMap<u32, TpLookupTable>,
}

impl TraceLookupTable {
    fn from_perf_path(perf_path: &Path) -> Result<Self> {
        let mut tp_layer_samples: TpLayerSamples = HashMap::new();

        for row in read_csv::<TableCsvRow>(perf_path)? {
            tp_layer_samples
                .entry(row.tp)
                .or_default()
                .entry(row.name)
                .or_default()
                .entry(row.key_0)
                .or_default()
                .entry(row.key_1)
                .or_default()
                .record(row.latency);
        }

        let mut tps = tp_layer_samples.keys().copied().collect::<Vec<_>>();
        tps.sort_unstable();
        tps.dedup();

        let mut tp_tables = HashMap::new();
        for tp in tps {
            let layer_latencies_s = tp_layer_samples
                .remove(&tp)
                .unwrap_or_default()
                .into_iter()
                .map(|(name, samples)| {
                    let samples = samples
                        .into_iter()
                        .map(|(key_0, inner)| {
                            let inner = inner
                                .into_iter()
                                .map(|(key_1, latency)| {
                                    latency.mean().map(|latency| (key_1, latency))
                                })
                                .collect::<Result<BTreeMap<_, _>>>()?;
                            Ok((key_0, inner))
                        })
                        .collect::<Result<BTreeMap<_, _>>>()?;
                    LookupTable2D::from_samples(samples, &format!("layer '{name}'"))
                        .map(|table| (name, table))
                })
                .collect::<Result<HashMap<_, _>>>()?;

            tp_tables.insert(tp, TpLookupTable { layer_latencies_s });
        }

        if tp_tables.is_empty() {
            bail!("native table {} is empty", perf_path.display());
        }

        Ok(Self { tp_tables })
    }

    fn lookup(&self, tp_size: u32, layer_name: &str, key: LookupKey) -> Result<f64> {
        let tp_table = self
            .tp_tables
            .get(&tp_size)
            .ok_or_else(|| anyhow!("unknown tp {}", tp_size))?;
        tp_table.lookup(layer_name, key)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TableLookupComputeSimulator {
    perf_dir: PathBuf,

    #[serde(skip_deserializing, default)]
    tp_size: Option<u32>,

    #[serde(skip_deserializing, default)]
    table: Option<TraceLookupTable>,
}

crate::register_compute_simulator!(TableLookupComputeSimulator, "table_lookup");

impl TableLookupComputeSimulator {
    fn lookup_params(
        layer: &Layer,
        batch: &[RequestState],
        context: ComputeContext,
    ) -> Result<LookupKey> {
        let keys = BatchKeys::from_batch(batch);
        let key = match layer {
            Layer::Attention { .. } => LookupKey {
                key_0: keys.attention_memory,
                key_1: keys.attention_compute,
            },
            Layer::Sampler { .. } => LookupKey {
                key_0: 0,
                key_1: keys.lm_heads,
            },
            Layer::Gemm { name, .. } if name == "lm_head" => LookupKey {
                key_0: 0,
                key_1: keys.lm_heads,
            },
            Layer::Embedding { .. }
            | Layer::LayerNorm { .. }
            | Layer::Gemm { .. }
            | Layer::Rope { .. }
            | Layer::Act { .. } => LookupKey {
                key_0: 0,
                key_1: keys.tokens,
            },
            Layer::Moe { .. } => LookupKey {
                key_0: keys.tokens,
                key_1: context
                    .moe_activated_experts
                    .ok_or_else(|| anyhow!("moe layer requires activated expert context"))?,
            },
            Layer::AllReduce { .. } => bail!("all-reduce layer cannot be simulated as compute"),
        };
        Ok(key)
    }
}

impl Module for TableLookupComputeSimulator {
    fn resolve_path(&mut self, base_dir: &Path) -> Result<()> {
        if self.perf_dir.is_relative() {
            self.perf_dir = base_dir.join(&self.perf_dir);
        }
        Ok(())
    }
}

impl ComputeSimulator for TableLookupComputeSimulator {
    fn init(&mut self, parallel: ParallelStrategy) -> Result<()> {
        self.tp_size = Some(parallel.tp);
        self.table = Some(TraceLookupTable::from_perf_path(&self.perf_dir)?);
        Ok(())
    }

    fn simulate(
        &self,
        layer: &Layer,
        batch: &[RequestState],
        context: ComputeContext,
    ) -> Result<f64> {
        let table = self
            .table
            .as_ref()
            .ok_or_else(|| anyhow!("compute simulator must be initialized before use"))?;
        let tp_size = self
            .tp_size
            .ok_or_else(|| anyhow!("compute simulator tp_size is not initialized"))?;
        let key = Self::lookup_params(layer, batch, context)?;
        table.lookup(tp_size, layer.name(), key)
    }
}

fn validate_latency(latency: f64, layer_name: &str) -> Result<()> {
    if !latency.is_finite() || latency < 0.0 {
        bail!("computed invalid latency for layer '{layer_name}'");
    }
    Ok(())
}

fn read_csv<T>(path: &Path) -> Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
{
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = ReaderBuilder::new().trim(csv::Trim::All).from_reader(file);
    reader
        .into_deserialize()
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("failed to read {}", path.display()))
}

fn lookup_bounds(keys: &[u64], query: u64) -> (usize, usize) {
    debug_assert!(!keys.is_empty());
    if keys.len() == 1 {
        return (0, 0);
    }

    match keys.binary_search(&query) {
        Ok(index) => (index, index),
        Err(0) => (0, 0),
        Err(index) if index >= keys.len() => (keys.len() - 2, keys.len() - 1),
        Err(index) => (index - 1, index),
    }
}

fn linear_interpolate(x0: u64, y0: f64, x1: u64, y1: f64, query: u64) -> f64 {
    if x0 == x1 {
        return y0;
    }
    let span = (x1 - x0) as f64;
    let offset = (query.saturating_sub(x0)) as f64;
    y0 + (y1 - y0) * (offset / span)
}
