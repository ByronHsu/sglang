use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};

#[tokio::test]
async fn messages_count_tokens_is_forwarded_to_worker_unchanged() {
    let ctx = AppTestContext::new(vec![MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }])
    .await;
    let app = ctx.create_app().await;
    let payload = json!({
        "model": "mock-model",
        "system": "You are an XRD expert.",
        "messages": [{
            "role": "user",
            "content": "refine this pattern"
        }],
        "tools": [{
            "name": "RIETVELD_REFINEMENT",
            "description": "Run a refinement",
            "input_schema": {"type": "object", "properties": {}}
        }],
        "tool_choice": {"type": "auto"}
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages/count_tokens")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["input_tokens"], 3);
    assert_eq!(body["received"], payload);

    ctx.shutdown().await;
}
