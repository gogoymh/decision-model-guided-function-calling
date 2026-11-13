package dmcall

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Function-calling models over plain HTTP (net/http only, no vendor SDKs).
//
// Each provider turns (messages, tools, repair history) into one request and the response into a
// proposal: calls ([] for no call), or nil when the output holds no complete proposal (cut off at
// the token limit, filtered or refused). Keys come from options or environment variables and never
// appear in errors, strings or results.

// Generator proposes a candidate for an example after the given repair history. *Provider implements
// it; any type with this method can drive the loop.
type Generator interface {
	Generate(ctx context.Context, example Example, history []HistoryEntry) (*Proposal, error)
}

// Example is what a generator answers: the conversation and the available tools.
type Example struct {
	Messages []Message
	Tools    []Tool
}

// Proposal is one generator output.
type Proposal struct {
	// Calls is the candidate: nil when the output holds no complete proposal, empty for no call.
	Calls []Call
	// Content is the text the model wrote ("" if none).
	Content string
	// ParseErrors explain an unreadable output or malformed arguments.
	ParseErrors []string
	// Incomplete is why the output holds no complete proposal ("" when it is complete): one of the
	// Incomplete* reasons.
	Incomplete string
	// Provider is the provider's name (openai, anthropic, ...); RequestedModel the model asked for and
	// Model the one that answered ("" if the response does not say).
	Provider       string
	RequestedModel string
	Model          string
	Usage          Usage
	Latency        time.Duration
	RetryWait      time.Duration
	APICalls       int
}

// Why an output holds no complete proposal. Such a proposal is rejected and repaired.
const (
	IncompleteLength        = "length"
	IncompleteContentFilter = "content_filter"
	IncompleteRefusal       = "refusal"
	IncompletePromptFilter  = "prompt_filter"
	IncompleteMalformedCall = "malformed_call"
	IncompleteAbnormalStop  = "abnormal_stop"
)

// IncompleteReasons explain each Incomplete reason.
var IncompleteReasons = map[string]string{
	IncompleteLength:        "The response reached the output token limit before a complete proposal.",
	IncompleteContentFilter: "The provider's content filter stopped the response before a complete proposal.",
	IncompleteRefusal:       "The response was a refusal and contains no proposal.",
	IncompletePromptFilter:  "The provider's content filter rejected the request; no response was generated.",
	IncompleteMalformedCall: "The model produced a malformed function call.",
	IncompleteAbnormalStop:  "The response stopped abnormally before a complete proposal.",
}

// ProviderError is a sanitized provider failure. It never contains keys or response bodies.
type ProviderError struct {
	Message      string
	StatusCode   int
	RequestID    string
	Attempts     int
	ProviderCode string
}

func (e *ProviderError) Error() string { return e.Message }

type providerConfig struct {
	apiKey, baseURL *string
	client          *http.Client
	timeout         *time.Duration
	maxRetries      *int
	sleep           SleepFunc
	maxOutputTokens *int
	temperature     *float64
	systemPrompt    *string
	headers         map[string]string
	reasoningEffort *string
	parallelSet     bool
	parallel        *bool
	maxTokensField  *string
	path            *string
	endpoint        *string
	apiVersion      *string
	thinkingBudget  *int
}

type providerKind struct {
	class, name, family string // family: chat, anthropic or gemini
	defaultBaseURL      string
	baseURLEnv, keyEnv  []string
	needsKey            bool
	maxTokensField      string
	omitParallel        bool // parallel_tool_calls is sent only when set
}

var (
	kindOpenAI = &providerKind{class: "OpenAI", name: "openai", family: "chat",
		defaultBaseURL: "https://api.openai.com/v1", baseURLEnv: []string{"OPENAI_BASE_URL"},
		keyEnv: []string{"OPENAI_API_KEY"}, needsKey: true, maxTokensField: "max_completion_tokens"}
	kindOpenAICompatible = &providerKind{class: "OpenAICompatible", name: "openai-compatible", family: "chat",
		maxTokensField: "max_completion_tokens"}
	kindXAI = &providerKind{class: "XAI", name: "xai", family: "chat", defaultBaseURL: "https://api.x.ai/v1",
		keyEnv: []string{"XAI_API_KEY"}, needsKey: true, maxTokensField: "max_tokens"}
	kindOpenRouter = &providerKind{class: "OpenRouter", name: "openrouter", family: "chat",
		defaultBaseURL: "https://openrouter.ai/api/v1", keyEnv: []string{"OPENROUTER_API_KEY"}, needsKey: true,
		maxTokensField: "max_tokens"}
	kindMeta = &providerKind{class: "Meta", name: "meta", family: "chat", defaultBaseURL: "https://api.meta.ai/v1",
		keyEnv: []string{"MODEL_API_KEY"}, needsKey: true, maxTokensField: "max_completion_tokens", omitParallel: true}
	kindAzure = &providerKind{class: "AzureOpenAI", name: "azure", family: "chat",
		keyEnv: []string{"AZURE_OPENAI_API_KEY", "AZURE_API_KEY"}, needsKey: true,
		maxTokensField: "max_completion_tokens"}
	kindAnthropic = &providerKind{class: "Anthropic", name: "anthropic", family: "anthropic",
		defaultBaseURL: "https://api.anthropic.com/v1", baseURLEnv: []string{"ANTHROPIC_BASE_URL"},
		keyEnv: []string{"ANTHROPIC_API_KEY"}, needsKey: true}
	kindGemini = &providerKind{class: "Gemini", name: "gemini", family: "gemini",
		defaultBaseURL: "https://generativelanguage.googleapis.com/v1beta",
		keyEnv:         []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"}, needsKey: true}
)

// Provider is a function-calling model reached over HTTP. It is safe for concurrent use.
type Provider struct {
	kind            *providerKind
	model           string
	apiKey          string
	baseURL         string
	maxOutputTokens int // 0: none sent
	temperature     *float64
	systemPrompt    string
	timeout         time.Duration
	maxRetries      int
	headers         map[string]string
	client          *http.Client
	sleep           SleepFunc
	reasoningEffort string
	parallel        *bool
	maxTokensField  string
	path            string
	apiVersion      string
	thinkingBudget  *int
}

// NewOpenAI returns OpenAI Chat Completions. Key: WithAPIKey or OPENAI_API_KEY (base URL:
// OPENAI_BASE_URL, optional).
func NewOpenAI(model string, opts ...ProviderOption) (*Provider, error) {
	return newProvider(kindOpenAI, model, opts)
}

// NewOpenAICompatible returns any Chat Completions endpoint with function tools (vLLM, Ollama, LM
// Studio, Together, Groq, ...). WithBaseURL is required; no key is needed.
func NewOpenAICompatible(model string, opts ...ProviderOption) (*Provider, error) {
	return newProvider(kindOpenAICompatible, model, opts)
}

// NewXAI returns xAI Grok, OpenAI-compatible. Key: XAI_API_KEY.
func NewXAI(model string, opts ...ProviderOption) (*Provider, error) {
	return newProvider(kindXAI, model, opts)
}

// NewOpenRouter returns OpenRouter, one OpenAI-compatible endpoint for many models. Key: OPENROUTER_API_KEY.
func NewOpenRouter(model string, opts ...ProviderOption) (*Provider, error) {
	return newProvider(kindOpenRouter, model, opts)
}

// NewMeta returns the Meta Model API (Muse models such as muse-spark-1.3), OpenAI-compatible. Key:
// WithAPIKey or MODEL_API_KEY. Muse models are reasoning models and Meta rejects reasoning_effort
// "none" for them, so neither reasoning_effort nor parallel_tool_calls is sent unless you set it.
func NewMeta(model string, opts ...ProviderOption) (*Provider, error) {
	return newProvider(kindMeta, model, opts)
}

// NewAzureOpenAI returns an Azure OpenAI deployment, directly or behind an API Management gateway.
// Key: AZURE_OPENAI_API_KEY; endpoint: WithEndpoint or AZURE_OPENAI_ENDPOINT; version:
// WithAPIVersion or AZURE_OPENAI_API_VERSION.
func NewAzureOpenAI(deployment string, opts ...ProviderOption) (*Provider, error) {
	return newProvider(kindAzure, deployment, opts)
}

// NewAnthropic returns the Anthropic Messages API. Key: ANTHROPIC_API_KEY (base URL:
// ANTHROPIC_BASE_URL, optional).
func NewAnthropic(model string, opts ...ProviderOption) (*Provider, error) {
	return newProvider(kindAnthropic, model, opts)
}

// NewGemini returns the Google Gemini API (generateContent with function declarations). Key:
// GEMINI_API_KEY or GOOGLE_API_KEY.
func NewGemini(model string, opts ...ProviderOption) (*Provider, error) {
	return newProvider(kindGemini, strings.TrimPrefix(model, "models/"), opts)
}

var providerSpecs = map[string]func(string, ...ProviderOption) (*Provider, error){
	"openai": NewOpenAI, "azure": NewAzureOpenAI, "anthropic": NewAnthropic, "claude": NewAnthropic,
	"gemini": NewGemini, "google": NewGemini, "xai": NewXAI, "grok": NewXAI, "openrouter": NewOpenRouter,
	"meta": NewMeta,
}

// ProviderFromSpec builds a provider from "provider:model", such as "openai:gpt-5-nano",
// "anthropic:claude-sonnet-4-5", "gemini:gemini-2.5-flash", "xai:grok-4", "meta:muse-spark-1.3",
// "openrouter:meta/llama" or "azure:<deployment>". Use NewOpenAICompatible for any other endpoint.
func ProviderFromSpec(spec string, opts ...ProviderOption) (*Provider, error) {
	name, model, found := strings.Cut(spec, ":")
	if !found || model == "" {
		return nil, errors.New(`dmcall: use "provider:model", e.g. "openai:gpt-5-nano", "anthropic:claude-sonnet-4-5", ` +
			`"gemini:gemini-2.5-flash", "xai:grok-4", "meta:muse-spark-1.3" or "azure:<deployment>"`)
	}
	build, ok := providerSpecs[strings.ToLower(name)]
	if !ok {
		return nil, fmt.Errorf("dmcall: unknown provider %s; choose from anthropic, azure, claude, gemini, google, grok, "+
			"meta, openai, openrouter, xai, or use NewOpenAICompatible for any other endpoint", pyStr(name))
	}
	return build(model, opts...)
}

func envValue(names []string) string {
	for _, name := range names {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value
		}
	}
	return ""
}

var apiVersionPattern = regexp.MustCompile(`^[0-9]{4}-[0-9]{2}-[0-9]{2}(-preview)?$`)

func newProvider(kind *providerKind, model string, opts []ProviderOption) (*Provider, error) {
	config := &providerConfig{}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt.applyProvider(config); err != nil {
			return nil, err
		}
	}
	chat := kind.family == "chat"
	switch {
	case !chat && (config.reasoningEffort != nil || config.parallelSet || config.maxTokensField != nil ||
		config.path != nil):
		return nil, fmt.Errorf("dmcall: reasoning effort, parallel tool calls, the max tokens field and the path "+
			"apply only to OpenAI-compatible providers, not %s", kind.class)
	case kind != kindAzure && (config.endpoint != nil || config.apiVersion != nil):
		return nil, errors.New("dmcall: WithEndpoint and WithAPIVersion apply only to AzureOpenAI")
	case kind == kindAzure && config.baseURL != nil:
		return nil, errors.New("dmcall: use WithEndpoint for AzureOpenAI")
	case kind != kindGemini && config.thinkingBudget != nil:
		return nil, errors.New("dmcall: WithThinkingBudget applies only to Gemini")
	}
	p := &Provider{kind: kind, model: model, timeout: 120 * time.Second, maxRetries: 3, maxOutputTokens: 4096,
		systemPrompt: SystemPrompt, client: config.client, sleep: config.sleep, temperature: config.temperature,
		thinkingBudget: config.thinkingBudget, maxTokensField: kind.maxTokensField, path: "/chat/completions"}
	baseURL := ""
	if kind == kindAzure {
		baseURL = firstNonEmpty(deref(config.endpoint), envValue([]string{"AZURE_OPENAI_ENDPOINT", "AZURE_ENDPOINT"}))
		if baseURL == "" {
			return nil, errors.New("dmcall: AzureOpenAI needs WithEndpoint or AZURE_OPENAI_ENDPOINT")
		}
		p.apiVersion = firstNonEmpty(deref(config.apiVersion), envValue([]string{"AZURE_OPENAI_API_VERSION"}),
			"2024-12-01-preview")
		if !apiVersionPattern.MatchString(p.apiVersion) {
			return nil, errors.New("dmcall: the API version must look like 2024-12-01-preview")
		}
		if model == "." || model == ".." || strings.IndexFunc(model, func(r rune) bool { return r < 32 }) >= 0 {
			return nil, errors.New("dmcall: the deployment name contains invalid characters")
		}
	}
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("dmcall: %s needs a model name", kind.class)
	}
	if config.maxRetries != nil {
		if *config.maxRetries < 0 {
			return nil, errors.New("dmcall: max retries must be a nonnegative integer")
		}
		p.maxRetries = *config.maxRetries
	}
	if config.maxOutputTokens != nil {
		if *config.maxOutputTokens < 0 {
			return nil, errors.New("dmcall: max output tokens must be a positive integer, or 0 for none")
		}
		p.maxOutputTokens = *config.maxOutputTokens
	}
	if config.timeout != nil {
		if *config.timeout <= 0 {
			return nil, errors.New("dmcall: timeout must be positive")
		}
		p.timeout = *config.timeout
	}
	if config.systemPrompt != nil {
		if strings.TrimSpace(*config.systemPrompt) == "" {
			return nil, errors.New("dmcall: the system prompt must be nonempty text")
		}
		p.systemPrompt = *config.systemPrompt
	}
	p.apiKey = firstNonEmpty(deref(config.apiKey), envValue(kind.keyEnv))
	if kind.needsKey && p.apiKey == "" {
		return nil, fmt.Errorf("dmcall: %s needs an API key: pass WithAPIKey or set %s", kind.class,
			strings.Join(kind.keyEnv, " or "))
	}
	if strings.ContainsAny(p.apiKey, "\r\n") {
		return nil, errors.New("dmcall: API keys must be single-line strings")
	}
	if kind != kindAzure {
		baseURL = firstNonEmpty(deref(config.baseURL), envValue(kind.baseURLEnv), kind.defaultBaseURL)
	}
	checked, ok := checkBaseURL(baseURL)
	if !ok {
		return nil, fmt.Errorf("dmcall: %s needs an http(s) base_url without credentials, query or fragment", kind.class)
	}
	p.baseURL = checked
	if chat {
		p.reasoningEffort = deref(config.reasoningEffort)
		if !kind.omitParallel {
			p.parallel = new(bool) // false by default
		}
		if config.parallelSet {
			p.parallel = config.parallel
		}
		if config.maxTokensField != nil && *config.maxTokensField != "" {
			p.maxTokensField = *config.maxTokensField
		}
		if config.path != nil {
			if !strings.HasPrefix(*config.path, "/") {
				return nil, errors.New("dmcall: the path must start with '/'")
			}
			p.path = *config.path
		}
	}
	if len(config.headers) > 0 {
		p.headers = config.headers
	}
	if p.client == nil {
		p.client = newHTTPClient(nil)
	}
	if p.sleep == nil {
		p.sleep = sleepContext
	}
	return p, nil
}

// Name is the provider's name: openai, openai-compatible, xai, openrouter, meta, azure, anthropic or gemini.
func (p *Provider) Name() string { return p.kind.name }

// Model is the requested model (the deployment for Azure).
func (p *Provider) Model() string { return p.model }

// BaseURL is the API's base URL.
func (p *Provider) BaseURL() string { return p.baseURL }

// String describes the provider without its key.
func (p Provider) String() string {
	return fmt.Sprintf("%s(model=%s, base_url=%s)", p.kind.class, pyStr(p.model), pyStr(p.baseURL))
}

// GoString uses the same sanitized representation for fmt's %#v format.
func (p Provider) GoString() string { return p.String() }

// pyQuote is Python's urllib.parse.quote(value, safe=""): every byte outside A-Z a-z 0-9 _.-~ is
// percent-encoded.
func pyQuote(value string) string {
	var b strings.Builder
	for i := 0; i < len(value); i++ {
		c := value[i]
		if c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || strings.IndexByte("_.-~", c) >= 0 {
			b.WriteByte(c)
		} else {
			fmt.Fprintf(&b, "%%%02X", c)
		}
	}
	return b.String()
}

// URL is the endpoint every request goes to.
func (p *Provider) URL() string {
	switch {
	case p.kind == kindAzure:
		return p.baseURL + "/openai/deployments/" + pyQuote(p.model) + "/chat/completions?api-version=" + p.apiVersion
	case p.kind.family == "anthropic":
		return p.baseURL + "/messages"
	case p.kind.family == "gemini":
		return p.baseURL + "/models/" + pyQuote(p.model) + ":generateContent"
	}
	return p.baseURL + p.path
}

func (p *Provider) requestHeaders() http.Header {
	header := http.Header{}
	for key, value := range p.headers {
		header.Set(key, value)
	}
	switch {
	case p.kind == kindAzure:
		header.Set("api-key", p.apiKey)
	case p.kind.family == "anthropic":
		header.Set("x-api-key", p.apiKey)
		header.Set("anthropic-version", "2023-06-01")
	case p.kind.family == "gemini":
		header.Set("x-goog-api-key", p.apiKey)
	case p.apiKey != "":
		header.Set("Authorization", "Bearer "+p.apiKey)
	}
	header.Set("Content-Type", "application/json")
	return header
}

// Requests ---------------------------------------------------------------------------------------------

// functionsOf returns each tool's function definition: name, description and (if set) parameters.
func functionsOf(tools []Tool) ([]*Object, error) {
	out := make([]*Object, 0, len(tools))
	for _, tool := range tools {
		function := obj("name", tool.Name, "description", tool.Description)
		if tool.Parameters != nil {
			parameters, err := toValue(tool.Parameters)
			if err != nil {
				return nil, err
			}
			function.Set("parameters", Clone(parameters))
		}
		out = append(out, function)
	}
	return out, nil
}

func parametersOf(function *Object) any {
	if parameters, ok := function.Get("parameters"); ok {
		return parameters
	}
	return emptyParameters()
}

// Payload returns the request body for an example after the given repair history.
func (p *Provider) Payload(example Example, history []HistoryEntry) (*Object, error) {
	messages, err := MessagesWithHistory(example.Messages, history)
	if err != nil {
		return nil, err
	}
	functions, err := functionsOf(example.Tools)
	if err != nil {
		return nil, err
	}
	switch p.kind.family {
	case "anthropic":
		return p.anthropicPayload(messages, functions)
	case "gemini":
		return p.geminiPayload(messages, functions)
	}
	conversation := []any{obj("role", "system", "content", p.systemPrompt)}
	for _, message := range messages {
		conversation = append(conversation, obj("role", message.Role, "content", message.Content))
	}
	payload := obj("model", p.model, "messages", conversation)
	if len(functions) > 0 { // a request without functions carries no tool settings
		declared := make([]any, len(functions))
		for i, function := range functions {
			declared[i] = obj("type", "function", "function", function)
		}
		payload.Set("tools", declared)
		payload.Set("tool_choice", "auto")
		if p.parallel != nil {
			payload.Set("parallel_tool_calls", *p.parallel)
		}
	}
	if p.maxOutputTokens > 0 {
		payload.Set(p.maxTokensField, int64(p.maxOutputTokens))
	}
	if p.reasoningEffort != "" {
		payload.Set("reasoning_effort", p.reasoningEffort)
	}
	if p.temperature != nil {
		payload.Set("temperature", *p.temperature)
	}
	return payload, nil
}

func (p *Provider) anthropicPayload(messages []Message, functions []*Object) (*Object, error) {
	system := []string{p.systemPrompt}
	conversation := []any{}
	for _, message := range messages {
		switch message.Role {
		case "system", "developer":
			system = append(system, message.Content)
		case "user", "assistant":
			conversation = append(conversation, obj("role", message.Role, "content", message.Content))
		default:
			return nil, errors.New("dmcall: Anthropic accepts only user and assistant conversation messages")
		}
	}
	tokens := p.maxOutputTokens
	if tokens == 0 {
		tokens = 4096
	}
	payload := obj("model", p.model, "system", strings.Join(system, "\n\n"), "messages", conversation,
		"max_tokens", int64(tokens))
	if len(functions) > 0 {
		declared := make([]any, len(functions))
		for i, function := range functions {
			declared[i] = obj("name", function.Value("name"), "description", function.Value("description"),
				"input_schema", parametersOf(function))
		}
		payload.Set("tools", declared)
		payload.Set("tool_choice", obj("type", "auto", "disable_parallel_tool_use", true))
	}
	if p.temperature != nil {
		payload.Set("temperature", *p.temperature)
	}
	return payload, nil
}

func (p *Provider) geminiPayload(messages []Message, functions []*Object) (*Object, error) {
	system := []string{p.systemPrompt}
	contents := []any{}
	for _, message := range messages {
		switch message.Role {
		case "system", "developer":
			system = append(system, message.Content)
		case "user", "assistant":
			role := "user"
			if message.Role == "assistant" {
				role = "model"
			}
			contents = append(contents, obj("role", role, "parts", []any{obj("text", message.Content)}))
		default:
			return nil, errors.New("dmcall: Gemini accepts only user and assistant conversation messages")
		}
	}
	payload := obj("systemInstruction", obj("parts", []any{obj("text", strings.Join(system, "\n\n"))}),
		"contents", contents)
	if len(functions) > 0 {
		declared := make([]any, len(functions))
		for i, function := range functions {
			declared[i] = obj("name", function.Value("name"), "description", function.Value("description"),
				"parametersJsonSchema", parametersOf(function))
		}
		payload.Set("tools", []any{obj("functionDeclarations", declared)})
		payload.Set("toolConfig", obj("functionCallingConfig", obj("mode", "AUTO")))
	}
	config := NewObject()
	if p.maxOutputTokens > 0 {
		config.Set("maxOutputTokens", int64(p.maxOutputTokens))
	}
	if p.temperature != nil {
		config.Set("temperature", *p.temperature)
	}
	if p.thinkingBudget != nil {
		config.Set("thinkingConfig", obj("thinkingBudget", int64(*p.thinkingBudget)))
	}
	if config.Len() > 0 {
		payload.Set("generationConfig", config)
	}
	return payload, nil
}

// Sending ----------------------------------------------------------------------------------------------

func transportKind(err error) string {
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return "Timeout"
	}
	return "TransportError"
}

func providerRetryDelay(header http.Header, attempt int) time.Duration {
	fallback := math.Pow(2, float64(attempt))
	delay := fallback
	if value := header.Get("Retry-After"); value != "" {
		parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
		if err == nil {
			delay = parsed
		}
	}
	if math.IsNaN(delay) || math.IsInf(delay, 0) {
		return seconds(math.Min(fallback, 30))
	}
	return seconds(math.Max(0, math.Min(delay, 30)))
}

var errorCodePattern = regexp.MustCompile(`^[A-Za-z0-9_.-]{1,64}$`)

func providerErrorCode(body []byte) string {
	value, err := ParseJSON(body)
	if err != nil {
		return ""
	}
	data, ok := value.(*Object)
	if !ok {
		return ""
	}
	failure, ok := data.Value("error").(*Object)
	if !ok {
		return ""
	}
	var code any
	if text, isText := failure.Value("code").(string); isText {
		code = text
	} else if kind := failure.Value("type"); truthy(kind) {
		code = kind
	} else {
		code = failure.Value("status")
	}
	if text, ok := code.(string); ok && errorCodePattern.MatchString(text) {
		return text
	}
	return ""
}

func (p *Provider) filteredRequest(status int, code string) bool {
	return p.kind.family == "chat" && status == 400 && (code == "content_filter" || code == "invalid_prompt")
}

func (p *Provider) wait(ctx context.Context, delay time.Duration) (time.Duration, error) {
	started := time.Now()
	err := p.sleep(ctx, delay)
	return time.Since(started), err
}

// errProtocol marks a response that does not follow the provider's protocol.
var errProtocol = errors.New("protocol")

// Generate sends one request for example after the given repair history and decodes the proposal.
func (p *Provider) Generate(ctx context.Context, example Example, history []HistoryEntry) (*Proposal, error) {
	payload, err := p.Payload(example, history)
	if err != nil {
		return nil, err
	}
	body, err := encodeJSON(payload, compactStyle)
	if err != nil {
		return nil, err
	}
	started := time.Now()
	var waited time.Duration
	attempts := 0
	var reply *httpReply
	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		attempts++
		reply, err = exchange(ctx, p.client, p.timeout, http.MethodPost, p.URL(), p.requestHeaders(), []byte(body))
		if err != nil {
			if ctx.Err() != nil {
				return nil, ctx.Err()
			}
			if attempt == p.maxRetries {
				return nil, &ProviderError{Message: fmt.Sprintf("%s transport failed after %d attempts (%s).",
					p.kind.name, attempts, transportKind(err)), Attempts: attempts}
			}
			spent, err := p.wait(ctx, seconds(math.Min(math.Pow(2, float64(attempt)), 8)))
			waited += spent
			if err != nil {
				return nil, err
			}
			continue
		}
		if (reply.status == 408 || reply.status == 429 || reply.status >= 500) && attempt < p.maxRetries {
			spent, err := p.wait(ctx, providerRetryDelay(reply.header, attempt))
			waited += spent
			if err != nil {
				return nil, err
			}
			continue
		}
		break
	}
	latency := time.Since(started) - waited
	if latency < 0 {
		latency = 0
	}
	proposal := &Proposal{Provider: p.kind.name, RequestedModel: p.model, Usage: Usage{}, Latency: latency,
		RetryWait: waited, APICalls: attempts}
	ids := []string{"x-request-id", "request-id", "apim-request-id", "x-goog-request-id"}
	fail := func(message string) error {
		return &ProviderError{Message: message, StatusCode: reply.status, RequestID: requestID(reply.header, ids, p.apiKey),
			Attempts: attempts}
	}
	if reply.status < 200 || reply.status >= 300 {
		code := providerErrorCode(reply.body)
		if !p.filteredRequest(reply.status, code) {
			detail := ""
			if code != "" {
				detail = " (" + code + ")"
			}
			e := fail(fmt.Sprintf("%s HTTP %d%s; check the model, key and endpoint.", p.kind.name, reply.status, detail))
			e.(*ProviderError).ProviderCode = code
			return nil, e
		}
		proposal.ParseErrors = []string{IncompleteReasons[IncompletePromptFilter]}
		proposal.Incomplete = IncompletePromptFilter
		return proposal, nil
	}
	value, err := ParseJSON(reply.body)
	if err != nil {
		return nil, fail(fmt.Sprintf("The %s response did not match its protocol.", p.kind.name))
	}
	data, ok := value.(*Object)
	if !ok {
		return nil, fail("The response was not a JSON object.")
	}
	decoded, err := p.decode(data)
	if err == nil {
		proposal.Usage, err = p.usage(data)
	}
	if err != nil {
		if errors.Is(err, errProtocol) {
			return nil, fail(fmt.Sprintf("The %s response did not match its protocol.", p.kind.name))
		}
		return nil, fail(err.Error())
	}
	proposal.Calls, proposal.ParseErrors, proposal.Content, proposal.Incomplete =
		decoded.calls, decoded.errors, decoded.content, decoded.incomplete
	if model, ok := data.Value("model").(string); ok && model != "" {
		proposal.Model = model
	} else if !truthy(data.Value("model")) {
		proposal.Model, _ = data.Value("modelVersion").(string)
	}
	return proposal, nil
}

// Decoding ----------------------------------------------------------------------------------------------

type decoded struct {
	calls      []Call
	errors     []string
	content    string
	incomplete string
}

// decodeError is a provider error message raised while decoding.
type decodeError string

func (e decodeError) Error() string { return string(e) }

func incomplete(reason string) decoded {
	return decoded{errors: []string{IncompleteReasons[reason]}, incomplete: reason}
}

func complete(calls []Call, errs []string, text []string) decoded {
	if errs == nil {
		errs = []string{}
	}
	return decoded{calls: calls, errors: errs, content: strings.Join(text, "\n")}
}

// appendCall adds a call; malformed arguments stay a rejected, repairable call, never a no call.
func appendCall(calls []Call, errs []string, name any, arguments any, encoded bool) ([]Call, []string, error) {
	text, ok := name.(string)
	if !ok || text == "" {
		return nil, nil, decodeError("The response contained an invalid function name.")
	}
	if encodedText, ok := arguments.(string); encoded && ok {
		if strings.TrimSpace(encodedText) == "" {
			arguments = NewObject()
		} else if parsed, err := ParseJSON([]byte(encodedText)); err != nil {
			return append(calls, Call{Name: text}), append(errs, "Function arguments are not valid JSON."), nil
		} else {
			arguments = parsed
		}
	}
	if arguments == nil {
		arguments = NewObject()
	}
	object, ok := arguments.(*Object)
	if !ok {
		return append(calls, Call{Name: text}), append(errs, "Function arguments must be a JSON object."), nil
	}
	if !finite(object) {
		return append(calls, Call{Name: text}), append(errs, "Function arguments must contain finite JSON values."), nil
	}
	return append(calls, Call{Name: text, Arguments: object.Clone()}), errs, nil
}

func (p *Provider) decode(data *Object) (decoded, error) {
	switch p.kind.family {
	case "anthropic":
		return decodeAnthropic(data)
	case "gemini":
		return decodeGemini(data)
	}
	return decodeChat(data)
}

func decodeChat(data *Object) (decoded, error) {
	if data.Value("error") != nil {
		return decoded{}, decodeError("The response contained a provider error.")
	}
	choices, ok := data.Value("choices").([]any)
	if !ok || len(choices) == 0 {
		return decoded{}, decodeError("The response did not contain a choices list.")
	}
	choice, ok := choices[0].(*Object)
	if !ok {
		return decoded{}, decodeError("The response did not contain a choices list.")
	}
	if finish, _ := choice.Value("finish_reason").(string); finish == IncompleteLength || finish == IncompleteContentFilter {
		return incomplete(finish), nil
	}
	message, ok := choice.Value("message").(*Object)
	role, hasRole := message.Get("role")
	if !ok || (hasRole && role != "assistant") {
		return decoded{}, decodeError("The response did not contain an assistant message.")
	}
	if truthy(message.Value("refusal")) {
		return incomplete(IncompleteRefusal), nil
	}
	var calls []Call
	var errs []string
	calls = []Call{}
	native := message.Value("tool_calls")
	if !truthy(native) {
		native = []any{}
	}
	items, ok := native.([]any)
	if !ok {
		return decoded{}, decodeError("The response tool_calls must be a list.")
	}
	for _, item := range items {
		call, ok := item.(*Object)
		function, isFunction := call.Value("function").(*Object)
		if !ok || !isFunction {
			return decoded{}, decodeError("The response contained an invalid function-call object.")
		}
		arguments, present := function.Get("arguments")
		if !present {
			arguments = "{}"
		}
		var err error
		if calls, errs, err = appendCall(calls, errs, function.Value("name"), arguments, true); err != nil {
			return decoded{}, err
		}
	}
	// Legacy Chat Completions returns one function_call instead of tool_calls.
	if len(items) == 0 && message.Value("function_call") != nil {
		function, ok := message.Value("function_call").(*Object)
		if !ok {
			return decoded{}, decodeError("The response contained an invalid function-call object.")
		}
		arguments, present := function.Get("arguments")
		if !present {
			arguments = "{}"
		}
		var err error
		if calls, errs, err = appendCall(calls, errs, function.Value("name"), arguments, true); err != nil {
			return decoded{}, err
		}
	}
	var text []string
	switch content := message.Value("content").(type) {
	case nil:
	case string:
		if content != "" {
			text = append(text, content)
		}
	default:
		return decoded{}, decodeError("The message content was not text.")
	}
	return complete(calls, errs, text), nil
}

func decodeAnthropic(data *Object) (decoded, error) {
	if data.Value("type") == "error" || data.Value("error") != nil {
		return decoded{}, decodeError("The response contained a provider error.")
	}
	blocks, ok := data.Value("content").([]any)
	if data.Value("role") != "assistant" || !ok {
		return decoded{}, decodeError("The Anthropic response was not an assistant message.")
	}
	reason, _ := data.Value("stop_reason").(string)
	if reason == "max_tokens" || reason == "model_context_window_exceeded" {
		return incomplete(IncompleteLength), nil
	}
	details, _ := data.Value("stop_details").(*Object)
	if reason == "refusal" || (details != nil && details.Value("type") == "refusal") {
		return incomplete(IncompleteRefusal), nil
	}
	calls, errs, text := []Call{}, []string(nil), []string(nil)
	for _, item := range blocks {
		block, ok := item.(*Object)
		if !ok {
			return decoded{}, decodeError("An Anthropic content block was malformed.")
		}
		switch block.Value("type") {
		case "tool_use":
			var err error
			if calls, errs, err = appendCall(calls, errs, block.Value("name"), block.Value("input"), false); err != nil {
				return decoded{}, err
			}
		case "text":
			if value, ok := block.Value("text").(string); ok && value != "" {
				text = append(text, value)
			}
		}
	}
	return complete(calls, errs, text), nil
}

var geminiFiltered = map[string]bool{"SAFETY": true, "RECITATION": true, "BLOCKLIST": true, "PROHIBITED_CONTENT": true,
	"SPII": true, "IMAGE_SAFETY": true, "LANGUAGE": true}

func decodeGemini(data *Object) (decoded, error) {
	if data.Value("error") != nil {
		return decoded{}, decodeError("The response contained a provider error.")
	}
	candidates := data.Value("candidates")
	if !truthy(candidates) {
		if feedback, ok := data.Value("promptFeedback").(*Object); ok && truthy(feedback.Value("blockReason")) {
			return incomplete(IncompletePromptFilter), nil
		}
		return decoded{}, decodeError("The Gemini response had no candidates.")
	}
	list, ok := candidates.([]any)
	if !ok {
		return decoded{}, errProtocol
	}
	candidate, ok := list[0].(*Object)
	if !ok {
		return decoded{}, errProtocol
	}
	reason, _ := candidate.Value("finishReason").(string)
	switch {
	case reason == "MAX_TOKENS":
		return incomplete(IncompleteLength), nil
	case geminiFiltered[reason]:
		return incomplete(IncompleteContentFilter), nil
	case reason == "MALFORMED_FUNCTION_CALL" || reason == "UNEXPECTED_TOOL_CALL":
		return incomplete(IncompleteMalformedCall), nil
	}
	if value := candidate.Value("finishReason"); value != nil && reason != "STOP" && reason != "FINISH_REASON_UNSPECIFIED" {
		return incomplete(IncompleteAbnormalStop), nil
	}
	if _, ok := candidate.Value("content").(*Object); reason != "STOP" && !ok {
		return incomplete(IncompleteAbnormalStop), nil
	}
	var parts []any
	if content := candidate.Value("content"); truthy(content) {
		object, ok := content.(*Object)
		if !ok {
			return decoded{}, errProtocol
		}
		switch value := object.Value("parts").(type) {
		case []any:
			parts = value
		case *Object, string:
			if truthy(value) {
				return decoded{}, decodeError("A Gemini content part was malformed.")
			}
		default:
			if truthy(value) {
				return decoded{}, errProtocol
			}
		}
	}
	calls, errs, text := []Call{}, []string(nil), []string(nil)
	for _, item := range parts {
		part, ok := item.(*Object)
		if !ok {
			return decoded{}, decodeError("A Gemini content part was malformed.")
		}
		if rawCall, present := part.Get("functionCall"); present {
			call, ok := rawCall.(*Object)
			if !ok {
				return incomplete(IncompleteMalformedCall), nil
			}
			var err error
			if calls, errs, err = appendCall(calls, errs, call.Value("name"), call.Value("args"), false); err != nil {
				return decoded{}, err
			}
		} else if value, ok := part.Value("text").(string); ok && value != "" && !truthy(part.Value("thought")) {
			text = append(text, value)
		}
	}
	return complete(calls, errs, text), nil
}

// count is a token count from a response: absent is unknown, anything but a nonnegative integer is an error.
func count(v any) (int, bool, error) {
	switch x := v.(type) {
	case nil:
		return 0, false, nil
	case int64:
		if x >= 0 {
			return int(x), true, nil
		}
	}
	return 0, false, decodeError("The response contained invalid token usage.")
}

func (p *Provider) usage(data *Object) (Usage, error) {
	out := Usage{}
	put := func(key string, v any) error {
		n, known, err := count(v)
		if known {
			out[key] = n
		}
		return err
	}
	switch p.kind.family {
	case "anthropic":
		raw, ok := data.Value("usage").(*Object)
		if !ok {
			return out, nil
		}
		read, _, err := count(raw.Value("cache_read_input_tokens"))
		if err != nil {
			return nil, err
		}
		written, _, err := count(raw.Value("cache_creation_input_tokens"))
		if err != nil {
			return nil, err
		}
		base, known, err := count(raw.Value("input_tokens"))
		if err != nil {
			return nil, err
		}
		out["cached_input_tokens"] = read
		if err := put("output_tokens", raw.Value("output_tokens")); err != nil {
			return nil, err
		}
		if known {
			out["input_tokens"] = base + read + written
		}
	case "gemini":
		raw, ok := data.Value("usageMetadata").(*Object)
		if !ok {
			return out, nil
		}
		output, outputKnown, err := count(raw.Value("candidatesTokenCount"))
		if err != nil {
			return nil, err
		}
		thoughts, _, err := count(raw.Value("thoughtsTokenCount"))
		if err != nil {
			return nil, err
		}
		if err := put("input_tokens", raw.Value("promptTokenCount")); err != nil {
			return nil, err
		}
		if err := put("cached_input_tokens", raw.Value("cachedContentTokenCount")); err != nil {
			return nil, err
		}
		if outputKnown || thoughts != 0 {
			out["output_tokens"] = output + thoughts
		}
	default:
		raw, ok := data.Value("usage").(*Object)
		if !ok {
			return out, nil
		}
		if err := put("input_tokens", raw.Value("prompt_tokens")); err != nil {
			return nil, err
		}
		if details, ok := raw.Value("prompt_tokens_details").(*Object); ok {
			if err := put("cached_input_tokens", details.Value("cached_tokens")); err != nil {
				return nil, err
			}
		}
		if err := put("output_tokens", raw.Value("completion_tokens")); err != nil {
			return nil, err
		}
	}
	return out, nil
}
