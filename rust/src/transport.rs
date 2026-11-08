//! The HTTP layer. Every generator and decision model sends its requests through an
//! [`HttpTransport`], so tests can serve recorded responses without touching the network.
//!
//! With the default `ureq` feature, [`UreqTransport`] is a blocking client with rustls.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::error::Result;

/// The HTTP method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// GET (the PPLX Decider's provenance check).
    Get,
    /// POST with a JSON body.
    Post,
}

/// One HTTP request.
#[derive(Clone)]
pub struct HttpRequest {
    /// GET or POST.
    pub method: Method,
    /// The full URL.
    pub url: String,
    /// Header names and values, in the order they were added.
    pub headers: Vec<(String, String)>,
    /// The body (empty for GET).
    pub body: Vec<u8>,
    /// The time limit for the whole request.
    pub timeout: Duration,
}

impl HttpRequest {
    /// A header's value; the name is matched without regard to case.
    pub fn header(&self, name: &str) -> Option<&str> {
        find_header(&self.headers, name)
    }

    /// The body parsed as JSON.
    pub fn json(&self) -> Option<Value> {
        serde_json::from_slice(&self.body).ok()
    }
}

impl fmt::Debug for HttpRequest {
    // Headers may hold keys and the body may hold user data, so neither is shown.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpRequest").field("method", &self.method).field("url", &self.url).finish_non_exhaustive()
    }
}

/// One HTTP response.
#[derive(Clone, Default)]
pub struct HttpResponse {
    /// The status code.
    pub status: u16,
    /// Header names and values.
    pub headers: Vec<(String, String)>,
    /// The body.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// A response with a raw body.
    pub fn new(status: u16, body: impl Into<Vec<u8>>) -> Self {
        HttpResponse { status, headers: Vec::new(), body: body.into() }
    }

    /// A response with a JSON body.
    pub fn json(status: u16, body: &Value) -> Self {
        let mut response = HttpResponse::new(status, serde_json::to_vec(body).unwrap_or_default());
        response.headers.push(("content-type".into(), "application/json".into()));
        response
    }

    /// The same response with one more header.
    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }

    /// A header's value; the name is matched without regard to case.
    pub fn header(&self, name: &str) -> Option<&str> {
        find_header(&self.headers, name)
    }

    /// Whether the status is 2xx.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub(crate) fn parse_json(&self) -> Option<Value> {
        let text = std::str::from_utf8(&self.body).ok()?;
        crate::pyjson::loads(text).ok()
    }
}

impl fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpResponse").field("status", &self.status).finish_non_exhaustive()
    }
}

fn find_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers.iter().find(|(key, _)| key.eq_ignore_ascii_case(name)).map(|(_, value)| value.as_str())
}

/// A request that produced no response (connection, TLS or timeout failure). It is retried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportError {
    /// A short category, such as `Timeout` or `ConnectionFailed`; never a URL or a body.
    pub kind: String,
}

impl TransportError {
    /// A transport error of the given kind.
    pub fn new(kind: impl Into<String>) -> Self {
        TransportError { kind: kind.into() }
    }
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HTTP transport failed ({})", self.kind)
    }
}

impl std::error::Error for TransportError {}

/// Sends one HTTP request. Implement it to use another client or to serve recorded responses.
///
/// Any `Fn(&HttpRequest) -> Result<HttpResponse, TransportError>` closure is a transport.
pub trait HttpTransport: Send + Sync {
    /// Send the request and return the response, whatever its status.
    fn send(&self, request: &HttpRequest) -> std::result::Result<HttpResponse, TransportError>;
}

impl<F> HttpTransport for F
where
    F: Fn(&HttpRequest) -> std::result::Result<HttpResponse, TransportError> + Send + Sync,
{
    fn send(&self, request: &HttpRequest) -> std::result::Result<HttpResponse, TransportError> {
        self(request)
    }
}

/// A shared transport.
pub type SharedTransport = Arc<dyn HttpTransport>;

/// How long to wait before a retry. Tests pass a function that returns at once.
pub type Sleep = Arc<dyn Fn(Duration) + Send + Sync>;

pub(crate) fn default_sleep() -> Sleep {
    Arc::new(std::thread::sleep)
}

/// The default transport: [`UreqTransport`] with the `ureq` feature, otherwise an error.
pub(crate) fn default_transport(use_env_proxy: bool) -> Result<SharedTransport> {
    #[cfg(feature = "ureq")]
    {
        Ok(Arc::new(UreqTransport::new(use_env_proxy)))
    }
    #[cfg(not(feature = "ureq"))]
    {
        let _ = use_env_proxy;
        Err(crate::error::Error::invalid("No HTTP transport: enable the `ureq` feature or pass .transport(...)."))
    }
}

/// A blocking HTTP client (ureq with rustls). It reads proxies from the environment unless
/// created with `use_env_proxy = false`.
#[cfg(feature = "ureq")]
pub struct UreqTransport {
    agent: ureq::Agent,
}

#[cfg(feature = "ureq")]
impl UreqTransport {
    /// A client; `use_env_proxy` follows `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY`.
    pub fn new(use_env_proxy: bool) -> Self {
        let mut config = ureq::Agent::config_builder().http_status_as_error(false);
        if !use_env_proxy {
            config = config.proxy(None);
        }
        UreqTransport { agent: ureq::Agent::new_with_config(config.build()) }
    }
}

#[cfg(feature = "ureq")]
impl Default for UreqTransport {
    fn default() -> Self {
        UreqTransport::new(true)
    }
}

#[cfg(feature = "ureq")]
impl fmt::Debug for UreqTransport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("UreqTransport")
    }
}

#[cfg(feature = "ureq")]
fn transport_error(error: ureq::Error) -> TransportError {
    let kind = match error {
        ureq::Error::Timeout(_) => "Timeout",
        ureq::Error::HostNotFound => "HostNotFound",
        ureq::Error::ConnectionFailed => "ConnectionFailed",
        ureq::Error::Io(_) => "Io",
        ureq::Error::Protocol(_) => "Protocol",
        ureq::Error::BadUri(_) => "BadUri",
        ureq::Error::Http(_) => "Http",
        _ => "Transport",
    };
    TransportError::new(kind)
}

#[cfg(feature = "ureq")]
impl HttpTransport for UreqTransport {
    fn send(&self, request: &HttpRequest) -> std::result::Result<HttpResponse, TransportError> {
        let response = match request.method {
            Method::Post => {
                let mut builder = self.agent.post(&request.url);
                for (name, value) in &request.headers {
                    builder = builder.header(name.as_str(), value.as_str());
                }
                builder.config().timeout_global(Some(request.timeout)).build().send(&request.body[..])
            }
            Method::Get => {
                let mut builder = self.agent.get(&request.url);
                for (name, value) in &request.headers {
                    builder = builder.header(name.as_str(), value.as_str());
                }
                builder.config().timeout_global(Some(request.timeout)).build().call()
            }
        };
        let mut response = response.map_err(transport_error)?;
        let status = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .map(|(name, value)| (name.as_str().to_string(), value.to_str().unwrap_or_default().to_string()))
            .collect();
        let body = response.body_mut().with_config().limit(64 * 1024 * 1024).read_to_vec().map_err(transport_error)?;
        Ok(HttpResponse { status, headers, body })
    }
}
