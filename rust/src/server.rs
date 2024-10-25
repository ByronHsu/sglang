use actix_web::{get, post, web, App, HttpServer, HttpResponse, HttpRequest, Responder};
use bytes::Bytes;
use clap::Parser;
use rand::seq::SliceRandom;
use futures_util::StreamExt;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 3001)]
    port: u16,

    #[arg(long, value_delimiter = ',')]
    worker_urls: Vec<String>,
}


#[derive(Clone)]
struct AppState {
    worker_urls: Vec<String>,
    client: reqwest::Client,
}

#[get("/v1/models")]
async fn v1_model(
    data: web::Data<AppState>,
) -> impl Responder {
    let worker_url= match data.worker_urls.choose(&mut rand::thread_rng()) {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };
    // Use the shared client
    match data.client
        .get(&format!("{}/v1/models", worker_url))
        .send()
        .await 
    {
        Ok(res) => {
            let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
            // print the status
            println!("Worker URL: {}, Status: {}", worker_url, status);
            match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            }
        },
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

#[get("/get_model_info")]
async fn get_model_info(
    data: web::Data<AppState>,
) -> impl Responder {
    let worker_url= match data.worker_urls.choose(&mut rand::thread_rng()) {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };
    // Use the shared client
    match data.client
        .get(&format!("{}/get_model_info", worker_url))
        .send()
        .await 
    {
        Ok(res) => {
            let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
            // print the status
            println!("Worker URL: {}, Status: {}", worker_url, status);
            match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            }
        },
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

// no deser and ser, just forward and return
#[post("/generate")]
async fn generate(
    req: HttpRequest,
    body: Bytes,
    data: web::Data<AppState>,
) -> impl Responder {

    // create a router struct
    let worker_url= match data.worker_urls.choose(&mut rand::thread_rng()) {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };

    // Check if client requested streaming
    let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
        .map(|v| v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
        .unwrap_or(false);

    let res = match data.client
        .post(&format!("{}/generate", worker_url))
        .header(
            "Content-Type", 
            req.headers()
                .get("Content-Type")
                .and_then(|h| h.to_str().ok())
                .unwrap_or("application/json")
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
        match res.bytes().await {
            Ok(body) => HttpResponse::build(status).body(body.to_vec()),
            Err(_) => HttpResponse::InternalServerError().finish(),
        } 
    } else {
        HttpResponse::build(status)
            .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
            .streaming(res.bytes_stream().map(|b| match b {
                Ok(b) => Ok::<_, actix_web::Error>(b),
                Err(_) => Err(actix_web::Error::from(actix_web::error::ErrorInternalServerError("Failed to read stream"))),
            }))
    }
}

pub async fn startup(host: String, port: u16, worker_urls: Vec<String>) -> std::io::Result<()> {
    // Create client once with configuration
    let client = reqwest::Client::builder()
        .build()
        .expect("Failed to create HTTP client");

    // Store both worker_urls and client in AppState
    let app_state = web::Data::new(AppState { 
        worker_urls,
        client,
    });

    println!("Starting server on {}:{}", host, port);
    println!("Worker URLs: {:?}", app_state.worker_urls);

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(generate)
            .service(v1_model)
            .service(get_model_info)
    })
    .bind((host, port))?
    .run()
    .await
}