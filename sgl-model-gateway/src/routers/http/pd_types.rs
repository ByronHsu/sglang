//! Types and utilities for the prefill-decode (PD) disaggregated router.

/// Custom error type for PD router operations
#[derive(Debug, thiserror::Error)]
pub enum PDRouterError {
    #[error("Worker already exists: {url}")]
    WorkerAlreadyExists { url: String },

    #[error("Worker not found: {url}")]
    WorkerNotFound { url: String },

    #[error("Lock acquisition failed: {operation}")]
    LockError { operation: String },

    #[error("Health check failed for worker: {url}")]
    HealthCheckFailed { url: String },

    #[error("Invalid worker configuration: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Timeout waiting for worker: {url}")]
    Timeout { url: String },
}

/// Construct a full API URL from a base URL and path.
pub fn api_path(url: &str, api_path: &str) -> String {
    if api_path.starts_with('/') {
        format!("{}{}", url, api_path)
    } else {
        format!("{}/{}", url, api_path)
    }
}

use serde::{Deserialize, Serialize};

use crate::protocols::generate::GenerateRequest;

/// Stage-specific data-parallel ranks for a prefill-decode request.
///
/// `routed_dp_rank` remains supported as the legacy single-rank contract.
/// When either stage-specific field is omitted, the PD router falls back to
/// that legacy rank for the corresponding stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PDRankRouting {
    pub prefill: Option<usize>,
    pub decode: Option<usize>,
    pub legacy: Option<usize>,
}

/// `/generate` request with optional asymmetric PD data-parallel ranks.
///
/// The wrapper keeps the public protocol extension local to the HTTP entry
/// point; downstream workers receive a normal `GenerateRequest` with only the
/// rank appropriate for their stage.
#[derive(Deserialize)]
pub struct PDGenerateRequest {
    #[serde(flatten)]
    pub request: GenerateRequest,
    #[serde(default)]
    pub routed_prefill_dp_rank: Option<usize>,
    #[serde(default)]
    pub routed_decode_dp_rank: Option<usize>,
    #[serde(default)]
    pub routed_dp_rank: Option<usize>,
}

impl PDGenerateRequest {
    pub fn rank_routing(&self) -> PDRankRouting {
        PDRankRouting {
            prefill: self.routed_prefill_dp_rank,
            decode: self.routed_decode_dp_rank,
            legacy: self.routed_dp_rank,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn test_pd_generate_request_extracts_stage_and_legacy_ranks() {
        let request: PDGenerateRequest = serde_json::from_value(json!({
            "text": "hello",
            "routed_prefill_dp_rank": 3,
            "routed_decode_dp_rank": 41,
            "routed_dp_rank": 7,
        }))
        .unwrap();

        assert_eq!(request.request.text.as_deref(), Some("hello"));
        assert_eq!(
            request.rank_routing(),
            PDRankRouting {
                prefill: Some(3),
                decode: Some(41),
                legacy: Some(7),
            }
        );
    }
}

/// Optimized bootstrap wrapper for single requests.
#[derive(Serialize)]
pub struct RequestWithBootstrap<'a, T: Serialize> {
    #[serde(flatten)]
    pub original: &'a T,
    pub bootstrap_host: String,
    pub bootstrap_port: Option<u16>,
    pub bootstrap_room: u64,
}

/// Optimized bootstrap wrapper for batch requests.
#[derive(Serialize)]
pub struct BatchRequestWithBootstrap<'a, T: Serialize> {
    #[serde(flatten)]
    pub original: &'a T,
    pub bootstrap_host: Vec<String>,
    pub bootstrap_port: Vec<Option<u16>>,
    pub bootstrap_room: Vec<u64>,
}

/// Generate a random bootstrap room ID.
pub fn generate_room_id() -> u64 {
    // Generate a value in the range [0, 2^63 - 1] to match Python's random.randint(0, 2**63 - 1)
    rand::random::<u64>() & (i64::MAX as u64)
}

/// PD-specific routing policies.
#[derive(Debug, Clone, PartialEq)]
pub enum PDSelectionPolicy {
    Random,
    PowerOfTwo,
    CacheAware {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
    },
    Bucket {
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        bucket_adjust_interval_secs: usize,
    },
}
