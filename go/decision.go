package dmcall

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Decision models: D in the paper.
//
// A decision model answers two yes/no questions about a candidate call and returns the probability
// of yes for each: the function question D_f (is this the right function, or is making no call
// right?) and the argument question D_a (are the arguments present, supported by the request and
// in the right parameters?). The argument question is not asked for a no call, and neither question
// is asked for an unreadable candidate; such a score is nil.
//
// Both decision models of the paper speak the same HTTP protocol (POST /v1/systemone): Jev,
// TypeSafe's hosted decision model (jev-1.13.0), and the PPLX Decider, the open-weight Perplexity
// Decider v1.1 27B served on your own machine. Only the request, the function schemas and the
// candidate are sent, never a reference answer.

// DecisionModel judges one candidate. *SystemOne implements it; any type with this method can be
// passed to WithDecisionModel.
type DecisionModel interface {
	// Judge runs the schema check and the applicable questions on calls (nil: unreadable, empty:
	// no call) and returns the verdicts and the feedback record the generator reads.
	Judge(ctx context.Context, messages []Message, tools []Tool, calls []Call) (*Judgment, error)
}

// Asker asks the decision model's questions over some protocol. Use NewDecisionModel to turn an
// Asker into a DecisionModel with the same schema check, thresholds and feedback record as Jev.
type Asker interface {
	// Ask answers questions ({"function_correct": ..., "arguments_correct": ...}) about state
	// ({"messages", "tools", "proposed_calls"}) with a probability of yes for each key.
	Ask(ctx context.Context, state, questions *Object) (*Answer, error)
}

// AskerFunc adapts a function to an Asker.
type AskerFunc func(ctx context.Context, state, questions *Object) (*Answer, error)

// Ask calls f.
func (f AskerFunc) Ask(ctx context.Context, state, questions *Object) (*Answer, error) {
	return f(ctx, state, questions)
}

// Answer is a decision model's reply to one request.
type Answer struct {
	// Probabilities holds the probability of yes for each question key.
	Probabilities map[string]float64
	// Model is the model that answered.
	Model string
	// Usage holds the token counts the model reported; an unknown count is absent.
	Usage Usage
	// APICalls counts HTTP attempts, RetryWait the time spent waiting between them.
	APICalls  int
	RetryWait time.Duration
}

// Usage holds token counts by name (input_tokens, cached_input_tokens, output_tokens). A count the
// service did not report is absent.
type Usage map[string]int

func (u Usage) add(other Usage) {
	for key, value := range other {
		u[key] += value
	}
}

func (u Usage) jsonValue() any {
	out := NewObject()
	for _, key := range []string{"input_tokens", "cached_input_tokens", "output_tokens"} {
		if value, ok := u[key]; ok {
			out.Set(key, int64(value))
		}
	}
	var rest []string
	for key := range u {
		if !out.Has(key) {
			rest = append(rest, key)
		}
	}
	sort.Strings(rest)
	for _, key := range rest {
		out.Set(key, int64(u[key]))
	}
	return out
}

// Judgment is a decision model's verdict on one candidate.
type Judgment struct {
	FunctionPass bool
	// ArgumentsPass is nil when the argument question was skipped (no call).
	ArgumentsPass    *bool
	ArgumentsSkipped bool
	// The probabilities of yes; nil for a question that was not asked.
	FunctionProbability  *float64
	ArgumentsProbability *float64
	// Accepted is the decision model's verdict with the schema check: both applicable checks pass.
	Accepted bool
	// Feedback is the record the generator reads after a rejection (before representation).
	Feedback     *Object
	SchemaErrors []SchemaError
	Usage        Usage
	// Model is the model that answered ("" if nothing was asked); RequestedModel the one requested.
	Model          string
	RequestedModel string
	Latency        time.Duration
	RetryWait      time.Duration
	APICalls       int
}

// DecisionModelError is a transport or protocol failure. It is never a rejection, and it never holds
// keys or bodies.
type DecisionModelError struct {
	Message    string
	Code       string // http_error, connection_failed, response_invalid, model_mismatch, ...
	StatusCode int
	RequestID  string
	Attempts   int
}

func (e *DecisionModelError) Error() string { return e.Message }

// RetryStatuses are the HTTP statuses a decision model request retries.
var RetryStatuses = map[int]bool{408: true, 429: true, 500: true, 502: true, 503: true, 504: true, 529: true}

// Jev and PPLX Decider constants.
const (
	JevDefaultModel   = "jev-1.13.0"
	JevDefaultBaseURL = "https://api.typesafe.ai"

	PPLXModelID                = "perplexity-ai/pplx-decider-v1.1-27b"
	PPLXModelRevision          = "14c9d90137962d300f3de95633f7b6535e80c5ce"
	PPLXCalibrationTemperature = 1.0087417621345625
	PPLXMaxInputTokens         = 8192
	// The checkpoint's server reports this inherited name and accepts "jev-latest" as the request model.
	PPLXServerModel = "autojev-qwen3.8-27b"
	PPLXWireModel   = "jev-latest"
)

const (
	jevFeedbackSource     = "jev_noul_and_deterministic_schema_validation"
	pplxFeedbackSource    = "perplexity_decider_noul_and_deterministic_schema_validation"
	genericFeedbackSource = "decision_model_noul_and_deterministic_schema_validation"
)

// SystemOne is a decision model reached over the /v1/systemone protocol (Jev, the PPLX Decider or
// any compatible server), or one built from your own Asker with NewDecisionModel. It is safe for
// concurrent use.
type SystemOne struct {
	name           string
	kind           string // "Jev", "PPLXDecider", "DecisionModel"
	feedbackSource string
	baseURL        string
	model          string
	apiKey         *string // pointer keeps a dereferenced value's debug formatting from revealing the key

	functionThreshold  float64
	argumentsThreshold float64
	timeout            time.Duration
	maxRetries         int
	client             *http.Client
	sleep              SleepFunc
	asker              Asker

	pplx            bool
	checkProvenance bool
	provenanceMu    sync.Mutex
	provenance      *Object
}

type decisionConfig struct {
	apiKey, baseURL, model *string
	threshold              *float64
	functionThreshold      *float64
	argumentsThreshold     *float64
	timeout                *time.Duration
	maxRetries             *int
	client                 *http.Client
	sleep                  SleepFunc
	checkProvenance        *bool
}

func decisionOptions(opts []DecisionOption) (*decisionConfig, error) {
	config := &decisionConfig{}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt.applyDecision(config); err != nil {
			return nil, err
		}
	}
	return config, nil
}

// NewJev returns TypeSafe's hosted decision model. The key comes from WithAPIKey or
// TYPESAFE_API_KEY; the model defaults to jev-1.13.0, the one used in the paper (TYPESAFE_MODEL or
// WithModel changes it), and the endpoint to https://api.typesafe.ai (TYPESAFE_BASE_URL or
// WithBaseURL). Both questions pass at 0.5 unless WithThreshold says otherwise.
func NewJev(opts ...DecisionOption) (*SystemOne, error) {
	config, err := decisionOptions(opts)
	if err != nil {
		return nil, err
	}
	if config.checkProvenance != nil {
		return nil, errors.New("dmcall: WithCheckProvenance applies only to the PPLX Decider")
	}
	key := ""
	if config.apiKey != nil {
		key = *config.apiKey
	}
	if key == "" {
		key = strings.TrimSpace(os.Getenv("TYPESAFE_API_KEY"))
	}
	if key == "" {
		return nil, errors.New("dmcall: Jev needs an API key: pass WithAPIKey or set TYPESAFE_API_KEY")
	}
	config.apiKey = &key
	baseURL := firstNonEmpty(deref(config.baseURL), os.Getenv("TYPESAFE_BASE_URL"), JevDefaultBaseURL)
	model := firstNonEmpty(deref(config.model), os.Getenv("TYPESAFE_MODEL"), JevDefaultModel)
	return newSystemOne("Jev", "Jev", jevFeedbackSource, baseURL, model, config, 60*time.Second, 3)
}

// NewPPLXDecider returns Perplexity Decider v1.1 27B, an open-weight decision model served on your
// own machine. baseURL is the origin of a server running the checkpoint with its native inference
// code, which serves /v1/systemone. Plain HTTP is accepted only for localhost; use an SSH tunnel
// or HTTPS otherwise. With WithCheckProvenance(true) the client first reads GET /provenance and
// stops unless the server reports the pinned checkpoint. No key is needed; the default timeout is
// 180 seconds with no retries.
func NewPPLXDecider(baseURL string, opts ...DecisionOption) (*SystemOne, error) {
	config, err := decisionOptions(opts)
	if err != nil {
		return nil, err
	}
	if config.model != nil {
		return nil, errors.New("dmcall: the PPLX Decider's model is fixed; WithModel does not apply")
	}
	if config.baseURL != nil {
		return nil, errors.New("dmcall: pass the PPLX Decider's base URL as its first argument")
	}
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return nil, errors.New("dmcall: PPLXDecider needs an http(s) base_url without credentials, query or fragment")
	}
	if parsed.Path != "" && parsed.Path != "/" {
		return nil, errors.New("dmcall: PPLXDecider base_url must be an origin without a path")
	}
	if parsed.Scheme == "http" {
		switch strings.ToLower(parsed.Hostname()) {
		case "localhost", "127.0.0.1", "::1":
		default:
			return nil, errors.New("dmcall: use HTTPS, or an SSH tunnel to localhost, for a remote PPLX Decider")
		}
	}
	if config.client == nil {
		// A local server is reached directly, never through an environment proxy.
		// Instrumentation may replace DefaultTransport with a RoundTripper wrapper.
		// A fresh transport still guarantees that environment proxies are bypassed.
		transport := &http.Transport{DialContext: (&net.Dialer{Timeout: 30 * time.Second,
			KeepAlive: 30 * time.Second}).DialContext, ForceAttemptHTTP2: true,
			MaxIdleConns: 100, IdleConnTimeout: 90 * time.Second,
			TLSHandshakeTimeout: 10 * time.Second, ExpectContinueTimeout: time.Second}
		if defaults, ok := http.DefaultTransport.(*http.Transport); ok {
			transport = defaults.Clone()
		}
		transport.Proxy = nil
		config.client = newHTTPClient(transport)
	}
	decider, err := newSystemOne("PPLX Decider", "PPLXDecider", pplxFeedbackSource, baseURL, PPLXWireModel, config,
		180*time.Second, 0)
	if err != nil {
		return nil, err
	}
	decider.pplx = true
	decider.checkProvenance = config.checkProvenance != nil && *config.checkProvenance
	return decider, nil
}

// NewSystemOne returns a decision model for any server that speaks the /v1/systemone protocol.
func NewSystemOne(baseURL, model string, opts ...DecisionOption) (*SystemOne, error) {
	config, err := decisionOptions(opts)
	if err != nil {
		return nil, err
	}
	if config.checkProvenance != nil {
		return nil, errors.New("dmcall: WithCheckProvenance applies only to the PPLX Decider")
	}
	if config.baseURL != nil || config.model != nil {
		return nil, errors.New("dmcall: pass the base URL and model to NewSystemOne as arguments")
	}
	return newSystemOne("Decision model", "DecisionModel", genericFeedbackSource, baseURL, model, config,
		60*time.Second, 3)
}

// NewDecisionModel builds a decision model from your own Asker. The schema check, the questions,
// the thresholds (0.5 unless set) and the feedback record are the same as Jev's; name appears in
// messages and source in the feedback records ("" for the generic source).
func NewDecisionModel(name, source string, asker Asker, opts ...DecisionOption) (*SystemOne, error) {
	if asker == nil {
		return nil, errors.New("dmcall: NewDecisionModel needs an Asker")
	}
	config, err := decisionOptions(opts)
	if err != nil {
		return nil, err
	}
	if config.apiKey != nil || config.baseURL != nil || config.client != nil || config.timeout != nil ||
		config.maxRetries != nil || config.checkProvenance != nil {
		return nil, errors.New("dmcall: a decision model with its own Asker takes only thresholds and a model name")
	}
	if strings.TrimSpace(name) == "" {
		name = "Decision model"
	}
	if source == "" {
		source = genericFeedbackSource
	}
	model := firstNonEmpty(deref(config.model), name)
	decision := &SystemOne{name: name, kind: "DecisionModel", feedbackSource: source, model: model, asker: asker}
	if err := decision.setThresholds(config); err != nil {
		return nil, err
	}
	return decision, nil
}

func deref[T any](p *T) T {
	var zero T
	if p == nil {
		return zero
	}
	return *p
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func checkThreshold(value float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
		return errors.New("dmcall: thresholds must be numbers between 0 and 1")
	}
	return nil
}

func (s *SystemOne) setThresholds(config *decisionConfig) error {
	threshold := 0.5
	if config.threshold != nil {
		threshold = *config.threshold
	}
	s.functionThreshold, s.argumentsThreshold = threshold, threshold
	if config.functionThreshold != nil {
		s.functionThreshold = *config.functionThreshold
	}
	if config.argumentsThreshold != nil {
		s.argumentsThreshold = *config.argumentsThreshold
	}
	if err := checkThreshold(s.functionThreshold); err != nil {
		return err
	}
	return checkThreshold(s.argumentsThreshold)
}

// checkBaseURL is an http(s) URL without credentials, query or fragment, without a trailing slash.
func checkBaseURL(value string) (string, bool) {
	trimmed := strings.TrimRight(strings.TrimSpace(value), "/")
	parsed, err := url.Parse(trimmed)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", false
	}
	return trimmed, true
}

func newSystemOne(name, kind, source, baseURL, model string, config *decisionConfig, timeout time.Duration,
	retries int) (*SystemOne, error) {
	s := &SystemOne{name: name, kind: kind, feedbackSource: source, timeout: timeout, maxRetries: retries,
		client: config.client, sleep: config.sleep}
	if err := s.setThresholds(config); err != nil {
		return nil, err
	}
	if config.maxRetries != nil {
		if *config.maxRetries < 0 {
			return nil, errors.New("dmcall: max retries must be a nonnegative integer")
		}
		s.maxRetries = *config.maxRetries
	}
	if config.timeout != nil {
		if *config.timeout <= 0 {
			return nil, errors.New("dmcall: timeout must be positive")
		}
		s.timeout = *config.timeout
	}
	if config.apiKey != nil {
		key := *config.apiKey
		if strings.TrimSpace(key) == "" || strings.ContainsAny(key, "\r\n") {
			return nil, errors.New("dmcall: API keys must be nonempty single-line strings")
		}
		s.apiKey = &key
	}
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("dmcall: %s needs a model name", name)
	}
	s.model = model
	checked, ok := checkBaseURL(baseURL)
	if !ok {
		return nil, fmt.Errorf("dmcall: %s needs an http(s) base_url without credentials, query or fragment", kind)
	}
	s.baseURL = checked
	if s.client == nil {
		s.client = newHTTPClient(nil)
	}
	if s.sleep == nil {
		s.sleep = sleepContext
	}
	return s, nil
}

// Name is the decision model's display name: Jev, PPLX Decider or Decision model.
func (s *SystemOne) Name() string { return s.name }

// Model is the requested model: jev-1.13.0 for Jev, the checkpoint ID for the PPLX Decider.
func (s *SystemOne) Model() string {
	if s.pplx {
		return PPLXModelID
	}
	return s.model
}

// BaseURL is the server origin ("" for a decision model with its own Asker).
func (s *SystemOne) BaseURL() string { return s.baseURL }

// FunctionThreshold is the pass mark of the function question.
func (s *SystemOne) FunctionThreshold() float64 { return s.functionThreshold }

// ArgumentsThreshold is the pass mark of the argument question.
func (s *SystemOne) ArgumentsThreshold() float64 { return s.argumentsThreshold }

// String describes the decision model without its key.
func (s *SystemOne) String() string {
	if s.pplx {
		return fmt.Sprintf("PPLXDecider(model=%s, revision=%s, base_url=%s, function_threshold=%s, arguments_threshold=%s)",
			pyStr(PPLXModelID), pyStr(PPLXModelRevision[:8]), pyStr(s.baseURL), pyFloat(s.functionThreshold),
			pyFloat(s.argumentsThreshold))
	}
	return fmt.Sprintf("%s(model=%s, base_url=%s, function_threshold=%s, arguments_threshold=%s)", s.kind,
		pyStr(s.model), pyStr(s.baseURL), pyFloat(s.functionThreshold), pyFloat(s.argumentsThreshold))
}

// GoString keeps API keys and internals out of fmt's %#v format.
func (s *SystemOne) GoString() string { return s.String() }

// HTTP ------------------------------------------------------------------------------------------------

func (s *SystemOne) secret() string { return deref(s.apiKey) }

func (s *SystemOne) headers() http.Header {
	header := http.Header{}
	if s.secret() != "" {
		header.Set("Authorization", "Bearer "+s.secret())
	}
	header.Set("Content-Type", "application/json")
	return header
}

var requestIDPattern = regexp.MustCompile(`^[A-Za-z0-9._:-]{1,128}$`)

// requestID is a narrow diagnostic ID, never an echoed credential.
func requestID(header http.Header, names []string, secret string) string {
	for _, name := range names {
		value := header.Get(name)
		if value != "" && requestIDPattern.MatchString(value) && (secret == "" || !strings.Contains(value, secret)) {
			return value
		}
	}
	return ""
}

func decisionRetryDelay(header http.Header, attempt int) time.Duration {
	if header != nil {
		if value := header.Get("Retry-After"); value != "" {
			delay := -1.0
			if seconds, err := strconv.ParseFloat(strings.TrimSpace(value), 64); err == nil {
				delay = seconds
			} else if stamp, err := http.ParseTime(value); err == nil {
				delay = time.Until(stamp).Seconds()
			}
			if !math.IsNaN(delay) && !math.IsInf(delay, 0) && delay >= 0 {
				return seconds(math.Min(delay, 30))
			}
		}
	}
	return seconds(math.Min(math.Pow(2, float64(attempt)), 30))
}

func seconds(value float64) time.Duration { return time.Duration(value * float64(time.Second)) }

// maxResponseBytes bounds how much of a response is read; a longer body fails to parse.
const maxResponseBytes = 64 << 20

// newHTTPClient is the default client: like httpx, it does not follow redirects.
func newHTTPClient(transport http.RoundTripper) *http.Client {
	return &http.Client{Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}}
}

type httpReply struct {
	status int
	header http.Header
	body   []byte
}

// exchange sends one request and reads the whole reply. The per-request timeout applies.
func exchange(ctx context.Context, client *http.Client, timeout time.Duration, method, target string,
	header http.Header, body []byte) (*httpReply, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	request, err := http.NewRequestWithContext(ctx, method, target, reader)
	if err != nil {
		return nil, err
	}
	request.Header = header.Clone()
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	data, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes))
	if err != nil {
		return nil, err
	}
	return &httpReply{status: response.StatusCode, header: response.Header, body: data}, nil
}

func (s *SystemOne) wait(ctx context.Context, delay time.Duration) (time.Duration, error) {
	started := time.Now()
	err := s.sleep(ctx, delay)
	return time.Since(started), err
}

func (s *SystemOne) httpError(reply *httpReply, attempts int) *DecisionModelError {
	e := &DecisionModelError{Message: fmt.Sprintf("%s HTTP %d; no judgment recorded.", s.name, reply.status),
		Code: "http_error", StatusCode: reply.status, Attempts: attempts,
		RequestID: requestID(reply.header, []string{"x-request-id", "request-id", "apim-request-id"}, s.secret())}
	if s.pplx && reply.status == 422 {
		if body, err := ParseJSON(reply.body); err == nil {
			if detail, ok := body.(*Object); ok {
				if text, _ := pyStrOf(detail.Value("detail")); strings.Contains(text, "8192-token limit") {
					e.Code = "context_limit_exceeded"
				}
			}
		}
	}
	return e
}

// pyStrOf is str(value) for a JSON value from a response.
func pyStrOf(v any) (string, bool) {
	if text, ok := v.(string); ok {
		return text, true
	}
	return pyRepr(v), false
}

// post sends a request with bounded retries; it returns the body, the attempts and the time spent waiting.
func (s *SystemOne) post(ctx context.Context, payload *Object) (*Object, int, time.Duration, error) {
	data, err := encodeJSON(payload, compactStyle)
	if err != nil {
		return nil, 0, 0, err
	}
	var waited time.Duration
	for attempt := 0; attempt <= s.maxRetries; attempt++ {
		reply, err := exchange(ctx, s.client, s.timeout, http.MethodPost, s.baseURL+"/v1/systemone", s.headers(),
			[]byte(data))
		if err != nil {
			if ctx.Err() != nil {
				return nil, attempt + 1, waited, ctx.Err()
			}
			if attempt < s.maxRetries {
				spent, err := s.wait(ctx, decisionRetryDelay(nil, attempt))
				waited += spent
				if err != nil {
					return nil, attempt + 1, waited, err
				}
				continue
			}
			return nil, attempt + 1, waited, &DecisionModelError{
				Message: fmt.Sprintf("%s connection failed after %d attempts.", s.name, attempt+1),
				Code:    "connection_failed", Attempts: attempt + 1}
		}
		if RetryStatuses[reply.status] && attempt < s.maxRetries {
			spent, err := s.wait(ctx, decisionRetryDelay(reply.header, attempt))
			waited += spent
			if err != nil {
				return nil, attempt + 1, waited, err
			}
			continue
		}
		if reply.status < 200 || reply.status >= 300 {
			return nil, attempt + 1, waited, s.httpError(reply, attempt+1)
		}
		value, err := ParseJSON(reply.body)
		if err != nil {
			return nil, attempt + 1, waited, &DecisionModelError{
				Message: s.name + " response was not valid JSON.", Code: "response_invalid"}
		}
		body, ok := value.(*Object)
		if !ok {
			return nil, attempt + 1, waited, &DecisionModelError{
				Message: s.name + " response must be a JSON object.", Code: "response_invalid"}
		}
		if s.pplx {
			if body, err = s.checkedPPLX(body); err != nil {
				return nil, attempt + 1, waited, err
			}
		}
		return body, attempt + 1, waited, nil
	}
	return nil, 0, waited, errors.New("dmcall: unreachable")
}

func (s *SystemOne) checkedPPLX(body *Object) (*Object, error) {
	if model := body.Value("model"); model != PPLXModelID && model != PPLXServerModel {
		return nil, &DecisionModelError{Message: "PPLX Decider response has an unexpected model ID.",
			Code: "model_mismatch"}
	}
	if revision, ok := body.Get("revision"); ok && revision != PPLXModelRevision {
		return nil, &DecisionModelError{Message: "PPLX Decider response has an unexpected model revision.",
			Code: "model_mismatch"}
	}
	checked := body.Clone()
	checked.Set("model", PPLXModelID)
	if !checked.Has("usage") { // missing usage is unknown, never zero
		checked.Set("usage", obj("input_tokens", nil, "output_tokens", nil))
	}
	return checked, nil
}

// Provenance reads GET /provenance from a PPLX Decider server and checks that it reports the pinned
// checkpoint, revision, attention mode, input limit and calibration temperature.
func (s *SystemOne) Provenance(ctx context.Context) (*Object, error) {
	if !s.pplx {
		return nil, errors.New("dmcall: provenance applies only to the PPLX Decider")
	}
	reply, err := exchange(ctx, s.client, s.timeout, http.MethodGet, s.baseURL+"/provenance", s.headers(), nil)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, &DecisionModelError{Message: "PPLX Decider provenance could not be reached.",
			Code: "connection_failed"}
	}
	if reply.status != 200 {
		return nil, &DecisionModelError{Message: "PPLX Decider provenance is unavailable.",
			Code: "provenance_unavailable", StatusCode: reply.status}
	}
	value, err := ParseJSON(reply.body)
	if err != nil {
		return nil, &DecisionModelError{Message: "PPLX Decider provenance was not valid JSON.", Code: "provenance_invalid"}
	}
	body, ok := value.(*Object)
	if !ok {
		return nil, &DecisionModelError{Message: "PPLX Decider provenance must be an object.", Code: "provenance_invalid"}
	}
	temperature, isNumber := number(body.Value("temperature"))
	if body.Value("model_id") != PPLXModelID || body.Value("revision") != PPLXModelRevision ||
		body.Value("native_attention") != "noncausal_full_attention" ||
		!Equal(body.Value("max_input_tokens"), int64(PPLXMaxInputTokens)) ||
		!isNumber || math.Abs(temperature-PPLXCalibrationTemperature) > 1e-12 {
		return nil, &DecisionModelError{Message: "The server does not run the pinned PPLX Decider checkpoint.",
			Code: "provenance_mismatch"}
	}
	return obj("model_id", PPLXModelID, "revision", PPLXModelRevision, "native_attention", "noncausal_full_attention",
		"max_input_tokens", int64(PPLXMaxInputTokens), "temperature", temperature), nil
}

// The protocol --------------------------------------------------------------------------------------

// Ask sends questions about state and returns the probabilities of yes, the model, usage and timing.
// For a decision model built with NewDecisionModel it calls that Asker.
func (s *SystemOne) Ask(ctx context.Context, state, questions *Object) (*Answer, error) {
	if s.asker != nil {
		return s.asker.Ask(ctx, state, questions)
	}
	if s.pplx && s.checkProvenance {
		s.provenanceMu.Lock()
		if s.provenance == nil {
			provenance, err := s.Provenance(ctx)
			if err != nil {
				s.provenanceMu.Unlock()
				return nil, err
			}
			s.provenance = provenance
		}
		s.provenanceMu.Unlock()
	}
	body, apiCalls, waited, err := s.post(ctx, obj("model", s.model, "state", state, "questions", questions))
	if err != nil {
		return nil, err
	}
	invalid := func(message string) error {
		return &DecisionModelError{Message: s.name + " " + message, Code: "response_invalid"}
	}
	answers, _ := body.Value("answers").(*Object)
	probabilities := map[string]float64{}
	for _, key := range questions.Keys() {
		answer, ok := answers.Value(key).(*Object)
		if !ok || answer.Value("type") != "noul" {
			return nil, invalid("response is missing the answer " + pyStr(key) + ".")
		}
		value := answer.Value("noul")
		p, isNumber := number(value)
		if !isNumber || math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
			return nil, invalid("returned an invalid probability for " + pyStr(key) + ".")
		}
		probabilities[key] = p
	}
	model, _ := body.Value("model").(string)
	if model == "" {
		return nil, invalid("response is missing the resolved model ID.")
	}
	usage, ok := body.Value("usage").(*Object)
	if !ok {
		return nil, invalid("response is missing token usage.")
	}
	counts := Usage{}
	for _, kind := range []string{"input_tokens", "output_tokens"} {
		switch count := usage.Value(kind).(type) {
		case nil:
		case int64:
			if count < 0 {
				return nil, invalid("response contains invalid token usage.")
			}
			counts[kind] = int(count)
		default:
			return nil, invalid("response contains invalid token usage.")
		}
	}
	return &Answer{Probabilities: probabilities, Model: model, Usage: counts, APICalls: apiCalls,
		RetryWait: waited}, nil
}

// Judge judges one candidate: the schema check S, both questions, the verdicts V and the feedback record.
func (s *SystemOne) Judge(ctx context.Context, messages []Message, tools []Tool, calls []Call) (*Judgment, error) {
	started := time.Now()
	compiled, err := compileTools(tools)
	if err != nil {
		return nil, err
	}
	local, err := compiled.check(callsValue(calls))
	if err != nil {
		return nil, err
	}
	skipped := calls != nil && len(calls) == 0
	var functionProbability, argumentsProbability *float64
	usage := Usage{"input_tokens": 0, "output_tokens": 0}
	var resolved string
	apiCalls := 0
	var waited time.Duration
	if local.WellFormed { // an unreadable candidate is not sent to the decision model
		observable := make([]any, len(messages))
		for i, message := range messages {
			observable[i] = message.jsonValue()
		}
		definitions, err := functionDefinitions(tools)
		if err != nil {
			return nil, err
		}
		proposed := make([]any, len(calls))
		for i, call := range calls {
			proposed[i] = call.Object()
		}
		state := obj("messages", observable, "tools", definitions, "proposed_calls", proposed)
		questions := obj("function_correct", FunctionQuestion.Object())
		if !skipped {
			questions.Set("arguments_correct", ArgumentQuestion.Object())
		}
		answer, err := s.Ask(ctx, state, questions)
		if err != nil {
			return nil, err
		}
		if answer == nil {
			return nil, fmt.Errorf("dmcall: %s returned no answer", s.name)
		}
		p, ok := answer.Probabilities["function_correct"]
		if !ok {
			return nil, &DecisionModelError{Message: s.name + " response is missing the answer 'function_correct'.",
				Code: "response_invalid"}
		}
		functionProbability = &p
		if !skipped {
			q, ok := answer.Probabilities["arguments_correct"]
			if !ok {
				return nil, &DecisionModelError{Message: s.name + " response is missing the answer 'arguments_correct'.",
					Code: "response_invalid"}
			}
			argumentsProbability = &q
		}
		resolved, usage = answer.Model, answer.Usage
		if usage == nil {
			usage = Usage{}
		}
		apiCalls, waited = answer.APICalls, answer.RetryWait
	}

	// V(p) is true iff p is not null and p >= threshold; S must pass too.
	functionPass := local.FunctionPass && functionProbability != nil && *functionProbability >= s.functionThreshold
	var argumentsPass *bool
	if !skipped {
		pass := local.ArgumentsPass && argumentsProbability != nil && *argumentsProbability >= s.argumentsThreshold
		argumentsPass = &pass
	}
	failed := []any{}
	if !functionPass {
		failed = append(failed, "function_selection")
	}
	if !skipped && !*argumentsPass {
		failed = append(failed, "arguments")
	}
	checks := obj(
		"function_selection", obj("pass", functionPass, "probability_yes", optionalFloat(functionProbability),
			"threshold", s.functionThreshold, "schema_pass", local.FunctionPass),
		"arguments", obj("skipped", skipped, "pass", optionalBool(argumentsPass),
			"probability_yes", optionalFloat(argumentsProbability), "threshold", s.argumentsThreshold,
			"schema_pass", local.ArgumentsPass),
	)
	feedback := obj("source", s.feedbackSource, "checks", checks, "failed_checks", failed,
		"schema_errors", schemaErrorsValue(local.Errors), "instruction", RepairInstruction)
	latency := time.Since(started) - waited
	if latency < 0 {
		latency = 0
	}
	return &Judgment{
		FunctionPass: functionPass, ArgumentsPass: argumentsPass, ArgumentsSkipped: skipped,
		FunctionProbability: functionProbability, ArgumentsProbability: argumentsProbability,
		Accepted: functionPass && (skipped || *argumentsPass), Feedback: feedback,
		SchemaErrors: append([]SchemaError{}, local.Errors...), Usage: usage, Model: resolved,
		RequestedModel: s.model, Latency: latency, RetryWait: waited, APICalls: apiCalls,
	}, nil
}

func optionalFloat(p *float64) any {
	if p == nil {
		return nil
	}
	return *p
}

func optionalBool(p *bool) any {
	if p == nil {
		return nil
	}
	return *p
}
