use anyhow::{bail, Result};
use serde::Deserialize;

use crate::model::Model;
use crate::module::Module;
use crate::types::{Layer, ParallelStrategy};

const FP_BYTES: u64 = 2;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Llama31Model {
    pub hidden_size: u64,
    pub intermediate_size: u64,
    pub max_position_embeddings: u64,
    pub num_attention_heads: u64,
    pub num_hidden_layers: u32,
    pub num_key_value_heads: u64,
    pub vocab_size: u64,

    #[serde(skip, default)]
    pre_block_layers: Vec<Layer>,

    #[serde(skip, default)]
    block_layers: Vec<Layer>,

    #[serde(skip, default)]
    post_block_layers: Vec<Layer>,

    #[serde(skip, default)]
    weight_bytes_per_device: u64,

    #[serde(skip, default)]
    kv_bytes_per_token_per_device: u64,
}

crate::register_model!(Llama31Model, "llama3_1");

impl Module for Llama31Model {}

impl Llama31Model {
    fn kv_dim(&self) -> u64 {
        let group = self.num_attention_heads / self.num_key_value_heads;
        self.hidden_size / group
    }

    fn head_dim(&self) -> u64 {
        self.hidden_size / self.num_attention_heads
    }

    fn tp_divisor(tp_size: u32) -> u64 {
        u64::from(tp_size.max(1))
    }

    fn tp_sync_bytes_per_token(&self) -> u64 {
        self.hidden_size * FP_BYTES
    }

    fn build_pre_block_layers(&self) -> Vec<Layer> {
        vec![Layer::Embedding {
            name: "embedding".to_string(),
            num_embeddings: self.vocab_size,
            embedding_dim: self.hidden_size,
        }]
    }

    fn build_dense_block_layers(&self, tp_size: u32) -> Vec<Layer> {
        let tp = Self::tp_divisor(tp_size);
        let kv_dim = self.kv_dim();
        let mut layers = vec![
            Layer::LayerNorm {
                name: "layernorm".to_string(),
                hidden_size: self.hidden_size,
            },
            Layer::Gemm {
                name: "qkv_projection".to_string(),
                m: self.hidden_size,
                n: (self.hidden_size + 2 * kv_dim) / tp,
            },
            Layer::Rope {
                name: "rope".to_string(),
                head_dim: self.head_dim(),
                num_heads: self.num_attention_heads / tp,
            },
            Layer::Attention {
                name: "attention".to_string(),
                num_q_heads: self.num_attention_heads / tp,
                num_kv_heads: self.num_key_value_heads / tp,
                head_dim: self.head_dim(),
            },
            Layer::Gemm {
                name: "o_projection".to_string(),
                m: self.hidden_size / tp,
                n: self.hidden_size,
            },
        ];
        if tp_size > 1 {
            layers.push(Layer::AllReduce {
                name: "all_reduce".to_string(),
                bytes_per_token: self.tp_sync_bytes_per_token(),
            });
        }
        layers.extend([
            Layer::LayerNorm {
                name: "layernorm".to_string(),
                hidden_size: self.hidden_size,
            },
            Layer::Gemm {
                name: "ffn1".to_string(),
                m: self.hidden_size,
                n: 2 * (self.intermediate_size / tp),
            },
            Layer::Act {
                name: "act".to_string(),
                width: self.intermediate_size / tp,
            },
            Layer::Gemm {
                name: "ffn2".to_string(),
                m: self.intermediate_size / tp,
                n: self.hidden_size,
            },
        ]);
        if tp_size > 1 {
            layers.push(Layer::AllReduce {
                name: "all_reduce".to_string(),
                bytes_per_token: self.tp_sync_bytes_per_token(),
            });
        }
        layers
    }

    fn build_post_block_layers(&self, tp_size: u32) -> Vec<Layer> {
        let tp = Self::tp_divisor(tp_size);
        vec![
            Layer::LayerNorm {
                name: "final_layernorm".to_string(),
                hidden_size: self.hidden_size,
            },
            Layer::Gemm {
                name: "lm_head".to_string(),
                m: self.hidden_size,
                n: self.vocab_size / tp,
            },
            Layer::Sampler {
                name: "sampler".to_string(),
                vocab_size: self.vocab_size,
            },
        ]
    }

    fn compute_weight_bytes_per_device(&self, tp_size: u32) -> u64 {
        let tp = Self::tp_divisor(tp_size);
        let kv_dim = self.kv_dim();
        let embed = (self.vocab_size / tp) * self.hidden_size;
        let q_proj = self.hidden_size * (self.hidden_size / tp);
        let k_proj = self.hidden_size * (kv_dim / tp);
        let v_proj = self.hidden_size * (kv_dim / tp);
        let o_proj = (self.hidden_size / tp) * self.hidden_size;
        let attn = q_proj + k_proj + v_proj + o_proj;
        let mlp = 2 * self.hidden_size * (self.intermediate_size / tp)
            + (self.intermediate_size / tp) * self.hidden_size;
        let layernorm = 2 * self.hidden_size;
        let block = attn + mlp + layernorm;
        let final_norm = self.hidden_size;
        let lm_head = self.hidden_size * (self.vocab_size / tp);
        FP_BYTES * (embed + u64::from(self.num_hidden_layers) * block + final_norm + lm_head)
    }

    fn compute_kv_bytes_per_token_per_device(&self, tp_size: u32) -> u64 {
        let tp = Self::tp_divisor(tp_size);
        2 * self.kv_dim() * u64::from(self.num_hidden_layers) * FP_BYTES / tp
    }
}

impl Model for Llama31Model {
    fn init(&mut self, parallel: ParallelStrategy) -> Result<()> {
        let tp_size = parallel.tp;
        if tp_size == 0 {
            bail!("tp_size must be positive");
        }
        if self.hidden_size == 0 {
            bail!("hidden_size must be positive");
        }
        if self.intermediate_size == 0 {
            bail!("intermediate_size must be positive");
        }
        if self.max_position_embeddings == 0 {
            bail!("max_position_embeddings must be positive");
        }
        if self.num_attention_heads == 0 {
            bail!("num_attention_heads must be positive");
        }
        if self.num_key_value_heads == 0 {
            bail!("num_key_value_heads must be positive");
        }
        if self.num_hidden_layers == 0 {
            bail!("num_hidden_layers must be positive");
        }
        if self.vocab_size == 0 {
            bail!("vocab_size must be positive");
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            bail!("hidden_size must be divisible by num_attention_heads");
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            bail!("num_attention_heads must be divisible by num_key_value_heads");
        }

        self.pre_block_layers = self.build_pre_block_layers();
        self.block_layers = self.build_dense_block_layers(tp_size);
        self.post_block_layers = self.build_post_block_layers(tp_size);
        self.weight_bytes_per_device = self.compute_weight_bytes_per_device(tp_size);
        self.kv_bytes_per_token_per_device = self.compute_kv_bytes_per_token_per_device(tp_size);
        Ok(())
    }

    fn weight_bytes_per_device(&self) -> u64 {
        self.weight_bytes_per_device
    }

    fn kv_bytes_per_token_per_device(&self) -> u64 {
        self.kv_bytes_per_token_per_device
    }

    fn pre_block_layers(&self) -> &[Layer] {
        &self.pre_block_layers
    }

    fn block_layers(&self) -> &[Layer] {
        &self.block_layers
    }

    fn post_block_layers(&self) -> &[Layer] {
        &self.post_block_layers
    }

    fn num_blocks(&self) -> u32 {
        self.num_hidden_layers
    }
}
