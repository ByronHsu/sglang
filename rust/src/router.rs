use crate::tree::Tree;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use bytes::Bytes;
use futures_util::{Stream, StreamExt, TryStreamExt};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::pin::Pin;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Debug)]
pub enum Router {
    RoundRobin {
        worker_urls: Vec<String>,
        current_index: AtomicUsize,
    },
    Random {
        worker_urls: Vec<String>,
    },
    CacheAware {
        /*
        Cache-Aware Load Balancing Router

        This router combines two strategies to optimize both cache utilization and request distribution:

        1. Cache-Aware Routing (Approximate Tree)
        2. Load Balancing (Shortest Queue)

        For each incoming request, the router chooses between these strategies:
        - With probability P: Uses cache-aware routing
        - With probability (1-P): Uses load balancing
        where P is configured via `cache_routing_prob`

        Strategy Details:

        1. Cache-Aware Routing (Approximate Tree)
        -------------------------------------------
        This strategy maintains an approximate radix tree for each worker based on request history,
        eliminating the need for direct cache state queries. The tree stores raw text characters
        instead of token IDs to avoid tokenization overhead.

        Process:
        a. For each request, find the worker with the highest prefix match
        b. If match rate > cache_threshold:
        Route to the worker with highest match (likely has relevant data cached)
        c. If match rate ≤ cache_threshold:
        Route to the worker with smallest tree size (most available cache capacity)
        d. Background maintenance:
        Periodically evict least recently used leaf nodes to prevent memory overflow

        2. Load Balancing (Shortest Queue)
        -------------------------------------------
        This strategy tracks pending request counts per worker and routes new requests
        to the least busy worker for optimal load distribution.

        Configuration Parameters:
        ------------------------
        1. cache_routing_prob: (float, 0.0 to 1.0)
        - 0.0: Exclusively use load balancing
        - 1.0: Exclusively use cache-aware routing
        - Between 0-1: Probability of using cache-aware routing vs load balancing

        2. cache_threshold: (float, 0.0 to 1.0)
        Minimum prefix match ratio to use highest-match routing.
        Below this threshold, routes to worker with most available cache space.

        3. eviction_interval_secs: (integer)
        Interval between LRU eviction cycles for the approximate trees.

        4. max_tree_size: (integer)
        Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
        during the next eviction cycle.
        */
        worker_urls: Vec<String>,
        tree: Arc<Mutex<Tree>>,
        running_queue: Arc<Mutex<HashMap<String, usize>>>,
        processed_queue: Arc<Mutex<HashMap<String, usize>>>,
        cache_threshold: f32,
        cache_routing_prob: f32,
        // 2D matrix of (user_id, worker_url) -> counter
        // Initialize with C for all pairs
        fairness_counter: Arc<Mutex<HashMap<String, HashMap<String, i32>>>>,
        fairness_fill_size: usize,
        enable_fairness: bool,
        _eviction_thread: Option<thread::JoinHandle<()>>, // Store thread handle
    },
}

#[derive(Debug)]
pub enum PolicyConfig {
    RandomConfig,
    RoundRobinConfig,
    CacheAwareConfig {
        cache_threshold: f32,
        cache_routing_prob: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        enable_fairness: bool,
        fairness_fill_size: usize,
    },
}

fn get_text_from_request(body: &Bytes, route: &str) -> String {
    // convert body to json
    let json = serde_json::from_slice::<serde_json::Value>(body).unwrap();

    if route == "generate" {
        // get the "text" field
        let text = json.get("text").and_then(|t| t.as_str()).unwrap_or("");
        return text.to_string();
    } else if route == "v1/chat/completions" {
        // get the messages field as raw text
        if let Some(messages) = json.get("messages") {
            // Convert messages back to a string, preserving all JSON formatting
            return serde_json::to_string(messages).unwrap_or_default();
        }
    } else if route == "v1/completions" {
        let prompt = json.get("prompt").and_then(|t| t.as_str()).unwrap_or("");
        return prompt.to_string();
    }

    return "".to_string();
}

fn get_uid_from_body(body: &Bytes) -> String {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()
        .and_then(|json| json.get("user").cloned())
        .and_then(|uid| uid.as_str().map(String::from))
        .unwrap_or_else(|| "default_uid".to_string())
}

impl Router {
    pub fn new(worker_urls: Vec<String>, policy_config: PolicyConfig) -> Self {
        match policy_config {
            PolicyConfig::RandomConfig => Router::Random { worker_urls },
            PolicyConfig::RoundRobinConfig => Router::RoundRobin {
                worker_urls,
                current_index: std::sync::atomic::AtomicUsize::new(0),
            },
            PolicyConfig::CacheAwareConfig {
                cache_threshold,
                cache_routing_prob,
                eviction_interval_secs,
                max_tree_size,
                enable_fairness,
                fairness_fill_size,
            } => {
                let mut running_queue = HashMap::new();
                for url in &worker_urls {
                    running_queue.insert(url.clone(), 0);
                }

                let mut processed_queue = HashMap::new();
                for url in &worker_urls {
                    processed_queue.insert(url.clone(), 0);
                }

                let tree = Arc::new(Mutex::new(Tree::new()));
                let running_queue = Arc::new(Mutex::new(running_queue));
                let processed_queue = Arc::new(Mutex::new(processed_queue));

                // Create background eviction thread
                let tree_clone = Arc::clone(&tree);
                let processed_queue_clone = Arc::clone(&processed_queue);
                let eviction_thread = thread::spawn(move || {
                    loop {
                        // Sleep for the specified interval
                        thread::sleep(Duration::from_secs(eviction_interval_secs));

                        let locked_tree_clone = tree_clone.lock().unwrap();
                        // Run eviction
                        locked_tree_clone.evict_tenant_data(max_tree_size);

                        // Print the process queue
                        let locked_processed_queue = processed_queue_clone.lock().unwrap();
                        println!("Processed Queue: {:?}", locked_processed_queue);
                    }
                });

                for url in &worker_urls {
                    tree.lock().unwrap().insert(&"".to_string(), url);
                }

                let fairness_counter = Arc::new(Mutex::new(HashMap::new()));

                Router::CacheAware {
                    worker_urls,
                    tree,
                    running_queue,
                    processed_queue,
                    cache_threshold,
                    cache_routing_prob,
                    fairness_counter,
                    enable_fairness,
                    fairness_fill_size,
                    _eviction_thread: Some(eviction_thread),
                }
            }
        }
    }

    pub fn get_first(&self) -> Option<String> {
        match self {
            Router::RoundRobin { worker_urls, .. }
            | Router::Random { worker_urls }
            | Router::CacheAware { worker_urls, .. } => {
                if worker_urls.is_empty() {
                    None
                } else {
                    Some(worker_urls[0].clone())
                }
            }
        }
    }

    pub async fn dispatch(
        &self,
        client: &reqwest::Client,
        req: HttpRequest,
        body: Bytes,
        route: &str,
    ) -> HttpResponse {
        let text = get_text_from_request(&body, route);
        // For Debug
        // println!("text: {:?}, route: {:?}", text, route);

        let worker_url = match self {
            Router::RoundRobin {
                worker_urls,
                current_index,
            } => {
                let idx = current_index
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |x| Some((x + 1) % worker_urls.len()),
                    )
                    .unwrap();

                worker_urls[idx].clone()
            }

            Router::Random { worker_urls } => {
                worker_urls[rand::random::<usize>() % worker_urls.len()].clone()
            }

            Router::CacheAware {
                worker_urls,
                tree,
                running_queue,
                processed_queue,
                cache_threshold,
                cache_routing_prob,
                fairness_counter,
                fairness_fill_size,
                enable_fairness,
                ..
            } => {
                let mut tree = tree.lock().unwrap();
                let mut running_queue = running_queue.lock().unwrap();

                // Generate a random float between 0 and 1 for probability check
                let sampled_p: f32 = rand::random();

                let selected_url = if *enable_fairness {

                    let user_id = get_uid_from_body(&body);

                    let mut fairness_counter = fairness_counter.lock().unwrap();
            
                    // Initialize counter for new user
                    if !fairness_counter.contains_key(&user_id) {
                        let mut worker_counters = HashMap::new();
                        for worker_url in worker_urls.iter() {
                            worker_counters.insert(worker_url.clone(), *fairness_fill_size as i32);
                        }
                        fairness_counter.insert(user_id.to_string(), worker_counters.clone());
                        
                        println!(
                            "[FAIRNESS] New user initialized. user_id: {}, initial_counters: {:?}",
                            user_id, worker_counters
                        );
                    }
            
                    let mut prefix_map: HashMap<String, String> = HashMap::new();
                    for worker_url in worker_urls.iter() {
                        let prefix = tree.prefix_match_tenant(&text, worker_url);
                        prefix_map.insert(worker_url.clone(), prefix);
                    }
            
                    let mut sorted_workers: Vec<_> = prefix_map.into_iter().collect();
                    sorted_workers.sort_by(|(_url1, prefix1), (_url2, prefix2)| {
                        prefix2.len().cmp(&prefix1.len())
                    });
            
                    let mut selected = None;
            
                    loop {
                        // Try to find worker with highest prefix match with available counters
                        for (worker_url, prefix) in &sorted_workers {
                            if let Some(worker_counters) = fairness_counter.get_mut(&user_id) {
                                if let Some(&count) = worker_counters.get(worker_url) {
                                    let deduction = text.chars().count();
                                    if count - deduction as i32 > 0 {
                                        selected = Some(worker_url.clone());
                                        let new_count = count.saturating_sub(deduction as i32);
                                        worker_counters.insert(worker_url.clone(), new_count);
                                        
                                        println!(
                                            "[FAIRNESS] Worker selected. user_id: {}, worker: {}, prefix_len: {}, prev_count: {}, deduction: {}, new_count: {}",
                                            user_id, worker_url, prefix.len(), count, deduction, new_count
                                        );
                                        break;
                                    }
                                }
                            }
                        }
            
                        // Refill counters if no available worker found
                        if selected.is_none() {
                            if let Some(worker_counters) = fairness_counter.get_mut(&user_id) {
                                println!(
                                    "[FAIRNESS] Refilling counters. user_id: {}, previous_counters: {:?}",
                                    user_id, worker_counters
                                );
                                
                                for worker_url in worker_urls.iter() {
                                    if let Some(&count) = worker_counters.get(worker_url) {
                                        let new_count = count + *fairness_fill_size as i32;
                                        worker_counters.insert(worker_url.clone(), new_count);
                                        
                                        println!(
                                            "[FAIRNESS] Worker refilled. user_id: {}, worker: {}, prev_count: {}, fill_size: {}, new_count: {}",
                                            user_id, worker_url, count, fairness_fill_size, new_count
                                        );
                                    }
                                }
                            }
                        } else {
                            break;
                        }
                    }
            
                    let selected_worker = selected.unwrap_or_else(|| {
                        println!(
                            "[FAIRNESS] WARNING: Fallback to default worker. user_id: {}, worker: {}",
                            user_id, &worker_urls[0]
                        );
                        worker_urls[0].clone()
                    });
            
                    // Log final counter state
                    if let Some(worker_counters) = fairness_counter.get(&user_id) {
                        println!(
                            "[FAIRNESS] Request complete. user_id: {}, selected_worker: {}, final_counters: {:?}",
                            user_id, selected_worker, worker_counters
                        );
                    }
            
                    selected_worker
                } else {
                    if sampled_p < *cache_routing_prob {
                        // Cache-aware routing logic
                        let (matched_text, matched_worker) = tree.prefix_match(&text);
                        let matched_rate =
                            matched_text.chars().count() as f32 / text.chars().count() as f32;

                        if matched_rate > *cache_threshold {
                            matched_worker.to_string()
                        } else {
                            tree.get_smallest_tenant()
                        }
                    } else {
                        // Shortest queue routing logic
                        running_queue
                            .iter()
                            .min_by_key(|(_url, &count)| count)
                            .map(|(url, _)| url.clone())
                            .unwrap_or_else(|| worker_urls[0].clone())
                    }
                };

                // Update running queue
                let count = running_queue.get_mut(&selected_url).unwrap();
                *count += 1;

                // Update processed queue
                let mut locked_processed_queue = processed_queue.lock().unwrap();
                let count = locked_processed_queue.get_mut(&selected_url).unwrap();
                *count += 1;

                // Update tree with the new request
                tree.insert(&text, &selected_url);

                selected_url
            }
        };

        let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
            .map(|v| v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
            .unwrap_or(false);

        let res = match client
            .post(format!("{}/{}", worker_url.clone(), route))
            .header(
                "Content-Type",
                req.headers()
                    .get("Content-Type")
                    .and_then(|h| h.to_str().ok())
                    .unwrap_or("application/json"),
            )
            .body(body.to_vec())
            .send()
            .await
        {
            Ok(res) => res,
            Err(_) => return HttpResponse::InternalServerError().finish(),
        };

        let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, get response first
            let response = match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            };

            // Then decrement running queue counter if using CacheAware
            if let Router::CacheAware { running_queue, .. } = self {
                if let Ok(mut queue) = running_queue.lock() {
                    if let Some(count) = queue.get_mut(&worker_url) {
                        *count = count.saturating_sub(1);
                    }
                }
            }

            response
        } else if let Router::CacheAware { running_queue, .. } = self {
            let running_queue = Arc::clone(running_queue);
            let worker_url = worker_url.clone();

            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(
                    res.bytes_stream()
                        .map_err(|_| {
                            actix_web::error::ErrorInternalServerError("Failed to read stream")
                        })
                        .inspect(move |bytes| {
                            let bytes = bytes.as_ref().unwrap();
                            if bytes
                                .as_ref()
                                .windows(12)
                                .any(|window| window == b"data: [DONE]")
                            {
                                let mut locked_queue = running_queue.lock().unwrap();
                                let count = locked_queue.get_mut(&worker_url).unwrap();
                                *count = count.saturating_sub(1);
                                // print
                                // println!("streaming is done!!")
                            }
                        }),
                )
        } else {
            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read stream")
                }))
        }
    }
}
