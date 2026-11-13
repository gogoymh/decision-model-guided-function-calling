//! DM-Call: Decision-Model-Guided Function Calling with Cumulative Verification Feedback.
//!
//! An LLM proposes a function call; a decision model checks whether the function and the
//! arguments are right before anything runs; a rejected call goes back to the same LLM with
//! every earlier candidate and its feedback, until a candidate passes or the repair limit is
//! reached.
//!
//! ```no_run
//! use dmcall::{DMCall, OpenAI, Tool};
//! use serde_json::json;
//!
//! # fn main() -> dmcall::Result<()> {
//! let weather = Tool::function(
//!     "get_weather",
//!     "Current weather for a city.",
//!     json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
//! )?
//! .with_handler(|arguments| Ok(json!({"city": arguments["city"], "temp": 21})));
//!
//! // OPENAI_API_KEY for the generator, TYPESAFE_API_KEY for the default decision model (Jev).
//! let agent = DMCall::builder(OpenAI::new("gpt-5-nano")?).tool(weather).build()?;
//! let result = agent.run("Is it warm in Seoul right now?")?;
//! println!("{result}");
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

mod core;
pub mod decision;
mod error;
pub mod feedback;
mod message;
pub mod prompts;
pub mod providers;
pub mod pyjson;
pub mod schema;
pub mod tools;
pub mod transport;

pub use crate::core::{
    verify, Attempt, CallOptions, CallResult, Checks, DMCall, DMCallBuilder, History, IntoCalls, IntoMessages,
    OnExhaustion, Probabilities, TraceStep, UsageSummary, Verdict, DEFAULT_MAX_REPAIRS, DEFAULT_THRESHOLD,
};
pub use crate::decision::{
    judge_candidate, Answer, DecisionModel, Jev, JevBuilder, Judgment, PPLXDecider, PPLXDeciderBuilder, SystemOne,
    SystemOneBuilder,
};
pub use crate::error::{BoxError, DecisionModelError, Error, ProviderError, Result};
pub use crate::feedback::{label, Feedback};
pub use crate::message::{Call, Message};
pub use crate::prompts::{ARGUMENT_QUESTION, FUNCTION_QUESTION, SYSTEM_PROMPT};
pub use crate::providers::{
    provider_from_spec, provider_from_spec_with, Anthropic, AzureOpenAI, Gemini, Generator, Meta, OpenAI,
    OpenAICompatible, OpenRouter, Proposal, Provider, ProviderBuilder, ProviderKind, ProviderOptions, Usage, XAI,
};
pub use crate::schema::{check_schema, SchemaCheck, SchemaError};
pub use crate::tools::{normalize_calls, schema_of, Handler, Tool};
pub use crate::transport::{HttpRequest, HttpResponse, HttpTransport, Method, SharedTransport, Sleep, TransportError};

#[cfg(feature = "ureq")]
pub use crate::transport::UreqTransport;

/// The crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
