package dmcall

import (
	"context"
	"errors"
	"math"
	"net/http"
	"time"
)

// Options are typed by what they configure. Some options configure more than one thing: WithAPIKey
// works for a provider and a decision model, WithThreshold for the client's default Jev and for a
// decision model, WithMaxRepairs for a client and for one request.

// Option configures a Client (New).
type Option interface{ applyClient(*clientConfig) error }

// CallOption configures one request (Call, Run, Verify).
type CallOption interface{ applyCall(*callConfig) error }

// ProviderOption configures a generator (NewOpenAI, NewAnthropic, ...).
type ProviderOption interface{ applyProvider(*providerConfig) error }

// DecisionOption configures a decision model (NewJev, NewPPLXDecider, ...).
type DecisionOption interface{ applyDecision(*decisionConfig) error }

// SleepFunc waits between retries. It returns early with ctx's error when ctx is done.
type SleepFunc func(ctx context.Context, delay time.Duration) error

func sleepContext(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

type clientConfig struct {
	tools             []Tool
	handlers          map[string]Handler
	decisionModel     DecisionModel
	threshold         *float64
	jevAPIKey         *string
	maxRepairs        *int
	checks            *string
	history           *string
	feedback          *string
	onExhaustion      *string
	preserve          *bool
	executeUnverified *bool
}

type callConfig struct {
	tools             *[]Tool
	handlers          map[string]Handler
	maxRepairs        *int
	onExhaustion      *string
	executeUnverified *bool
	other             string // the name of an option Verify does not take
}

// EndpointOption configures how a provider or a decision model reaches its service.
type EndpointOption struct {
	provider func(*providerConfig) error
	decision func(*decisionConfig) error
}

func (o EndpointOption) applyProvider(c *providerConfig) error { return o.provider(c) }
func (o EndpointOption) applyDecision(c *decisionConfig) error { return o.decision(c) }

// ThresholdOption sets the pass mark of both questions.
type ThresholdOption struct{ value float64 }

func (o ThresholdOption) applyClient(c *clientConfig) error {
	c.threshold = &o.value
	return nil
}

func (o ThresholdOption) applyDecision(c *decisionConfig) error {
	c.threshold = &o.value
	return nil
}

// LoopOption configures the loop for a client or for one request.
type LoopOption struct {
	client func(*clientConfig) error
	call   func(*callConfig) error
}

func (o LoopOption) applyClient(c *clientConfig) error { return o.client(c) }
func (o LoopOption) applyCall(c *callConfig) error     { return o.call(c) }

type clientOption func(*clientConfig) error

func (o clientOption) applyClient(c *clientConfig) error { return o(c) }

type providerOption func(*providerConfig) error

func (o providerOption) applyProvider(c *providerConfig) error { return o(c) }

type decisionOption func(*decisionConfig) error

func (o decisionOption) applyDecision(c *decisionConfig) error { return o(c) }

// Shared by providers and decision models -------------------------------------------------------------

// WithAPIKey sets the API key instead of reading it from the environment.
func WithAPIKey(key string) EndpointOption {
	return EndpointOption{
		provider: func(c *providerConfig) error { c.apiKey = &key; return nil },
		decision: func(c *decisionConfig) error { c.apiKey = &key; return nil },
	}
}

// WithBaseURL sets the service's base URL (an http(s) URL without credentials, query or fragment).
func WithBaseURL(baseURL string) EndpointOption {
	return EndpointOption{
		provider: func(c *providerConfig) error { c.baseURL = &baseURL; return nil },
		decision: func(c *decisionConfig) error { c.baseURL = &baseURL; return nil },
	}
}

// WithHTTPClient sets the HTTP client. Tests pass one with a custom RoundTripper.
func WithHTTPClient(client *http.Client) EndpointOption {
	return EndpointOption{
		provider: func(c *providerConfig) error { c.client = client; return nil },
		decision: func(c *decisionConfig) error { c.client = client; return nil },
	}
}

// WithTimeout sets the timeout of each HTTP attempt (providers 120 s, Jev 60 s, PPLX Decider 180 s).
func WithTimeout(timeout time.Duration) EndpointOption {
	return EndpointOption{
		provider: func(c *providerConfig) error { c.timeout = &timeout; return nil },
		decision: func(c *decisionConfig) error { c.timeout = &timeout; return nil },
	}
}

// WithMaxRetries sets how many HTTP retries follow the first attempt (3, or 0 for the PPLX Decider).
// It has nothing to do with DM-Call's repair limit.
func WithMaxRetries(retries int) EndpointOption {
	return EndpointOption{
		provider: func(c *providerConfig) error { c.maxRetries = &retries; return nil },
		decision: func(c *decisionConfig) error { c.maxRetries = &retries; return nil },
	}
}

// WithSleep replaces the wait between retries, for tests.
func WithSleep(sleep SleepFunc) EndpointOption {
	return EndpointOption{
		provider: func(c *providerConfig) error { c.sleep = sleep; return nil },
		decision: func(c *decisionConfig) error { c.sleep = sleep; return nil },
	}
}

// Client and decision model --------------------------------------------------------------------------

// WithThreshold sets the pass mark of both questions (0.5 in the paper). For a client it configures
// the default Jev; pass a configured decision model instead to change anything else.
func WithThreshold(threshold float64) ThresholdOption { return ThresholdOption{value: threshold} }

// Decision model only --------------------------------------------------------------------------------

// WithModel sets Jev's model (default jev-1.13.0 or TYPESAFE_MODEL), or the model name reported by
// a decision model built with NewDecisionModel.
func WithModel(model string) DecisionOption {
	return decisionOption(func(c *decisionConfig) error { c.model = &model; return nil })
}

// WithFunctionThreshold overrides the pass mark of the function question.
func WithFunctionThreshold(threshold float64) DecisionOption {
	return decisionOption(func(c *decisionConfig) error { c.functionThreshold = &threshold; return nil })
}

// WithArgumentsThreshold overrides the pass mark of the argument question.
func WithArgumentsThreshold(threshold float64) DecisionOption {
	return decisionOption(func(c *decisionConfig) error { c.argumentsThreshold = &threshold; return nil })
}

// WithCheckProvenance makes the PPLX Decider read GET /provenance once and stop unless the server
// reports the pinned checkpoint. It is off by default.
func WithCheckProvenance(check bool) DecisionOption {
	return decisionOption(func(c *decisionConfig) error { c.checkProvenance = &check; return nil })
}

// Client only ---------------------------------------------------------------------------------------

// WithDecisionModel sets the decision model (default: Jev with WithThreshold, 0.5).
func WithDecisionModel(model DecisionModel) Option {
	return clientOption(func(c *clientConfig) error {
		if model == nil {
			return errors.New("dmcall: WithDecisionModel needs a decision model")
		}
		c.decisionModel = model
		return nil
	})
}

// WithJevAPIKey sets the key of the default Jev (otherwise TYPESAFE_API_KEY).
func WithJevAPIKey(key string) Option {
	return clientOption(func(c *clientConfig) error { c.jevAPIKey = &key; return nil })
}

// WithChecks selects the verdicts used for acceptance and repair feedback: ChecksBoth
// (DM-Call), ChecksFunction (DM-Call-Function) or ChecksArguments (DM-Call-Argument).
// Both questions are sent for a readable call; a no call receives only the function question.
func WithChecks(checks string) Option {
	return clientOption(func(c *clientConfig) error { c.checks = &checks; return nil })
}

// WithHistory selects what the generator sees: HistoryCumulative (DM-Call) or HistoryLatest
// (DM-Call-Latest, only the latest rejection with its real round number).
func WithHistory(history string) Option {
	return clientOption(func(c *clientConfig) error { c.history = &history; return nil })
}

// WithFeedback selects the feedback representation: FeedbackHybrid, FeedbackNumeric or FeedbackText.
func WithFeedback(feedback string) Option {
	return clientOption(func(c *clientConfig) error { c.feedback = &feedback; return nil })
}

// WithPreserveInitialNoCall controls whether a readable initial no call is returned without
// verification or repair (true by default, the paper's rule).
func WithPreserveInitialNoCall(preserve bool) Option {
	return clientOption(func(c *clientConfig) error { c.preserve = &preserve; return nil })
}

// Client and one request ------------------------------------------------------------------------------

// WithTools sets the available tools. For one request it replaces the client's tools and their
// handlers: only the handlers of these tools and WithHandlers of that request apply.
func WithTools(tools ...Tool) LoopOption {
	return LoopOption{
		client: func(c *clientConfig) error { c.tools = append([]Tool(nil), tools...); return nil },
		call: func(c *callConfig) error {
			copied := append([]Tool{}, tools...)
			c.tools = &copied
			return nil
		},
	}
}

// WithHandlers adds handlers by tool name, for tools defined without one.
func WithHandlers(handlers map[string]Handler) LoopOption {
	merge := func(target *map[string]Handler) {
		if *target == nil {
			*target = map[string]Handler{}
		}
		for name, handler := range handlers {
			(*target)[name] = handler
		}
	}
	return LoopOption{
		client: func(c *clientConfig) error { merge(&c.handlers); return nil },
		call:   func(c *callConfig) error { merge(&c.handlers); c.other = "WithHandlers"; return nil },
	}
}

// WithMaxRepairs sets the repair limit r (3 in the paper): at most r+1 candidates.
func WithMaxRepairs(repairs int) LoopOption {
	return LoopOption{
		client: func(c *clientConfig) error { c.maxRepairs = &repairs; return nil },
		call:   func(c *callConfig) error { c.maxRepairs = &repairs; c.other = "WithMaxRepairs"; return nil },
	}
}

// WithOnExhaustion selects the final candidate when none is accepted: ExhaustionFirst (the paper's
// rule) or ExhaustionLast.
func WithOnExhaustion(rule string) LoopOption {
	return LoopOption{
		client: func(c *clientConfig) error { c.onExhaustion = &rule; return nil },
		call:   func(c *callConfig) error { c.onExhaustion = &rule; c.other = "WithOnExhaustion"; return nil },
	}
}

// WithExecuteUnverified makes Run execute the final candidate even when none was accepted (false by
// default). Call never executes anything.
func WithExecuteUnverified(execute bool) LoopOption {
	return LoopOption{
		client: func(c *clientConfig) error { c.executeUnverified = &execute; return nil },
		call: func(c *callConfig) error {
			c.executeUnverified = &execute
			c.other = "WithExecuteUnverified"
			return nil
		},
	}
}

// Providers only ------------------------------------------------------------------------------------------

// WithMaxOutputTokens sets the output token limit (4096 by default). Zero sends no limit, except to
// Anthropic, which requires one and receives 4096.
func WithMaxOutputTokens(tokens int) ProviderOption {
	return providerOption(func(c *providerConfig) error { c.maxOutputTokens = &tokens; return nil })
}

// WithTemperature sets the sampling temperature (not sent by default).
func WithTemperature(temperature float64) ProviderOption {
	return providerOption(func(c *providerConfig) error {
		if math.IsNaN(temperature) || math.IsInf(temperature, 0) {
			return errors.New("dmcall: temperature must be a finite number")
		}
		c.temperature = &temperature
		return nil
	})
}

// WithSystemPrompt replaces the generator's system prompt (SystemPrompt by default).
func WithSystemPrompt(prompt string) ProviderOption {
	return providerOption(func(c *providerConfig) error { c.systemPrompt = &prompt; return nil })
}

// WithHeaders adds HTTP headers to every request; the auth and Content-Type headers take precedence.
func WithHeaders(headers map[string]string) ProviderOption {
	return providerOption(func(c *providerConfig) error {
		if c.headers == nil {
			c.headers = map[string]string{}
		}
		for key, value := range headers {
			c.headers[key] = value
		}
		return nil
	})
}

// WithReasoningEffort sends reasoning_effort (OpenAI-compatible providers; the paper used "none").
func WithReasoningEffort(effort string) ProviderOption {
	return providerOption(func(c *providerConfig) error { c.reasoningEffort = &effort; return nil })
}

// WithParallelToolCalls sends parallel_tool_calls (OpenAI-compatible providers; false by default).
func WithParallelToolCalls(parallel bool) ProviderOption {
	return providerOption(func(c *providerConfig) error {
		c.parallelSet, c.parallel = true, &parallel
		return nil
	})
}

// WithoutParallelToolCalls leaves parallel_tool_calls out of the request.
func WithoutParallelToolCalls() ProviderOption {
	return providerOption(func(c *providerConfig) error {
		c.parallelSet, c.parallel = true, nil
		return nil
	})
}

// WithMaxTokensField names the output limit field (max_completion_tokens, or max_tokens for xAI and
// OpenRouter).
func WithMaxTokensField(field string) ProviderOption {
	return providerOption(func(c *providerConfig) error { c.maxTokensField = &field; return nil })
}

// WithPath sets the Chat Completions path of an OpenAI-compatible server ("/chat/completions").
func WithPath(path string) ProviderOption {
	return providerOption(func(c *providerConfig) error { c.path = &path; return nil })
}

// WithEndpoint sets the Azure OpenAI endpoint (otherwise AZURE_OPENAI_ENDPOINT).
func WithEndpoint(endpoint string) ProviderOption {
	return providerOption(func(c *providerConfig) error { c.endpoint = &endpoint; return nil })
}

// WithAPIVersion sets the Azure OpenAI API version (otherwise AZURE_OPENAI_API_VERSION or
// 2024-12-01-preview).
func WithAPIVersion(version string) ProviderOption {
	return providerOption(func(c *providerConfig) error { c.apiVersion = &version; return nil })
}

// WithThinkingBudget sends Gemini's thinkingConfig.thinkingBudget.
func WithThinkingBudget(budget int) ProviderOption {
	return providerOption(func(c *providerConfig) error { c.thinkingBudget = &budget; return nil })
}
