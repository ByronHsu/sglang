use pyo3::prelude::*;
mod server;

#[pyclass]
struct Router {
    host: String,
    port: u16,
    worker_urls: Vec<String>,
}

#[pymethods]
impl Router {
    #[new]
    fn new(host: String, port: u16, worker_urls: Vec<String>) -> Self {
        Router {
            host,
            port,
            worker_urls,
        }
    }

    fn start(&self) -> PyResult<()> {
        let host = self.host.clone();
        let port = self.port;
        let worker_urls = self.worker_urls.clone();

        actix_web::rt::System::new().block_on(async move {
            server::startup(host, port, worker_urls).await.unwrap();
        });

        Ok(())
    }
}

#[pymodule]
fn router(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Router>()?;
    Ok(())
}