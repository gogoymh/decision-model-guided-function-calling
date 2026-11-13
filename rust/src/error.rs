//! Errors. None of them ever holds an API key, a request body or a response body.

use std::fmt;

/// A boxed error returned by a tool handler.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// `std::result::Result` with [`Error`] as the default error.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Every failure DM-Call reports.
#[derive(Debug)]
pub enum Error {
    /// An invalid argument or configuration (Python's `ValueError` and `TypeError`).
    Invalid(String),
    /// A generator request failed; this is never an unreadable candidate.
    Provider(ProviderError),
    /// A decision-model request failed; this is never a rejection.
    DecisionModel(DecisionModelError),
    /// `run` found an accepted call whose tool has no handler.
    MissingHandler(String),
    /// A tool handler returned an error.
    Tool {
        /// The tool's name.
        name: String,
        /// What the handler returned.
        source: BoxError,
    },
}

impl Error {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Error::Invalid(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(message) => f.write_str(message),
            Error::Provider(error) => fmt::Display::fmt(error, f),
            Error::DecisionModel(error) => fmt::Display::fmt(error, f),
            Error::MissingHandler(name) => write!(
                f,
                "No handler for tool {}; attach one with Tool::with_handler or DMCallBuilder::handler.",
                crate::pyjson::repr_str(name)
            ),
            Error::Tool { name, source } => write!(f, "Tool {} failed: {}", crate::pyjson::repr_str(name), source),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Provider(error) => Some(error),
            Error::DecisionModel(error) => Some(error),
            Error::Tool { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl From<ProviderError> for Error {
    fn from(error: ProviderError) -> Self {
        Error::Provider(error)
    }
}

impl From<DecisionModelError> for Error {
    fn from(error: DecisionModelError) -> Self {
        Error::DecisionModel(error)
    }
}

/// A sanitized generator failure: an HTTP error, a transport failure or a response that does
/// not follow the provider's protocol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderError {
    /// What went wrong, without keys or bodies.
    pub message: String,
    /// The HTTP status, when there was a response.
    pub status_code: Option<u16>,
    /// A narrow request ID from the response headers, for support tickets.
    pub request_id: Option<String>,
    /// How many HTTP attempts were made.
    pub attempts: Option<u32>,
    /// The provider's short error code, such as `invalid_api_key`.
    pub provider_code: Option<String>,
}

impl ProviderError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        ProviderError {
            message: message.into(),
            status_code: None,
            request_id: None,
            attempts: None,
            provider_code: None,
        }
    }
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ProviderError {}

/// A decision-model transport or protocol failure. It is never a rejection of the candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionModelError {
    /// What went wrong, without keys or bodies.
    pub message: String,
    /// A stable code: `http_error`, `connection_failed`, `response_invalid`, `model_mismatch`,
    /// `context_limit_exceeded`, `provenance_unavailable`, `provenance_invalid` or
    /// `provenance_mismatch`.
    pub code: String,
    /// The HTTP status, when there was a response.
    pub status_code: Option<u16>,
    /// A narrow request ID from the response headers.
    pub request_id: Option<String>,
    /// How many HTTP attempts were made.
    pub attempts: Option<u32>,
}

impl DecisionModelError {
    pub(crate) fn new(message: impl Into<String>, code: &str) -> Self {
        DecisionModelError {
            message: message.into(),
            code: code.to_string(),
            status_code: None,
            request_id: None,
            attempts: None,
        }
    }
}

impl fmt::Display for DecisionModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DecisionModelError {}
