package dmcall

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

// DM-Call: propose a function call, verify it with a decision model, repair it with cumulative feedback.
//
// At step t the generator proposes a candidate c_t (one call, or no call). The candidate is accepted
// when it passes the schema check S and every decision-model question that applies to it (Eq. 2 of
// the paper). A rejected candidate and its feedback d_t join the history H, which the generator
// reads, oldest first and marked as not executed, before the unchanged request. The loop returns the
// first accepted candidate; if none of c_0..c_r is accepted it returns c_0. Nothing is executed while
// the loop runs.

// The paper's settings.
const (
	DefaultThreshold  = 0.5 // pass mark of both questions
	DefaultMaxRepairs = 3   // r* in the paper
)

// Checks: which decision-model questions decide acceptance.
const (
	ChecksBoth      = "both"      // DM-Call
	ChecksFunction  = "function"  // DM-Call-Function
	ChecksArguments = "arguments" // DM-Call-Argument
)

// History: which rejected candidates the generator sees.
const (
	HistoryCumulative = "cumulative" // DM-Call
	HistoryLatest     = "latest"     // DM-Call-Latest
)

// OnExhaustion: the final candidate when none is accepted.
const (
	ExhaustionFirst = "first" // the paper's rule
	ExhaustionLast  = "last"
)

// Client runs DM-Call: it proposes with a generator, verifies with a decision model and repairs with
// cumulative feedback. It is safe for concurrent use.
type Client struct {
	generator             Generator
	decisionModel         DecisionModel
	tools                 []Tool
	handlers              map[string]Handler
	maxRepairs            int
	checks                string
	history               string
	feedback              string
	onExhaustion          string
	preserveInitialNoCall bool
	executeUnverified     bool
}

func choice(name, value string, choices ...string) error {
	for _, c := range choices {
		if value == c {
			return nil
		}
	}
	quoted := make([]string, len(choices))
	for i, c := range choices {
		quoted[i] = pyStr(c)
	}
	return fmt.Errorf("dmcall: %s must be one of %s", name, strings.Join(quoted, ", "))
}

func checkRepairs(repairs int) error {
	if repairs < 0 {
		return errors.New("dmcall: max repairs must be a nonnegative integer")
	}
	return nil
}

// New returns a client. The defaults are the paper's settings: at most 3 repairs, both questions,
// the whole history, Hybrid feedback, the first candidate when none passes, and a readable initial
// no call returned without verification. Without WithDecisionModel the decision model is Jev
// (TYPESAFE_API_KEY or WithJevAPIKey) with WithThreshold (0.5) for both questions.
func New(generator Generator, opts ...Option) (*Client, error) {
	if generator == nil {
		return nil, errors.New("dmcall: New needs a generator")
	}
	config := &clientConfig{}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt.applyClient(config); err != nil {
			return nil, err
		}
	}
	c := &Client{generator: generator, maxRepairs: DefaultMaxRepairs, checks: ChecksBoth, history: HistoryCumulative,
		feedback: FeedbackHybrid, onExhaustion: ExhaustionFirst, preserveInitialNoCall: true}
	if config.maxRepairs != nil {
		if err := checkRepairs(*config.maxRepairs); err != nil {
			return nil, err
		}
		c.maxRepairs = *config.maxRepairs
	}
	set := func(target *string, value *string, name string, choices ...string) error {
		if value == nil {
			return nil
		}
		if err := choice(name, *value, choices...); err != nil {
			return err
		}
		*target = *value
		return nil
	}
	if err := set(&c.checks, config.checks, "checks", ChecksBoth, ChecksFunction, ChecksArguments); err != nil {
		return nil, err
	}
	if err := set(&c.history, config.history, "history", HistoryCumulative, HistoryLatest); err != nil {
		return nil, err
	}
	if err := set(&c.feedback, config.feedback, "feedback", FeedbackHybrid, FeedbackNumeric, FeedbackText); err != nil {
		return nil, err
	}
	if err := set(&c.onExhaustion, config.onExhaustion, "on_exhaustion", ExhaustionFirst, ExhaustionLast); err != nil {
		return nil, err
	}
	if config.preserve != nil {
		c.preserveInitialNoCall = *config.preserve
	}
	if config.executeUnverified != nil {
		c.executeUnverified = *config.executeUnverified
	}
	c.tools = config.tools
	c.handlers = config.handlers
	if _, _, err := toolset(c.tools, c.handlers); err != nil { // fail fast on a bad tool
		return nil, err
	}
	if config.decisionModel != nil {
		if config.jevAPIKey != nil || config.threshold != nil {
			return nil, errors.New("dmcall: configure the decision model itself (API key, threshold), not the client")
		}
		c.decisionModel = config.decisionModel
		return c, nil
	}
	jevOptions := []DecisionOption{WithThreshold(DefaultThreshold)}
	if config.threshold != nil {
		jevOptions[0] = WithThreshold(*config.threshold)
	}
	if config.jevAPIKey != nil {
		jevOptions = append(jevOptions, WithAPIKey(*config.jevAPIKey))
	}
	jev, err := NewJev(jevOptions...)
	if err != nil {
		return nil, err
	}
	c.decisionModel = jev
	return c, nil
}

// Generator is the generator the client proposes with.
func (c *Client) Generator() Generator { return c.generator }

// DecisionModel is the decision model the client verifies with.
func (c *Client) DecisionModel() DecisionModel { return c.decisionModel }

// MaxRepairs is the repair limit r.
func (c *Client) MaxRepairs() int { return c.maxRepairs }

// Checks is ChecksBoth, ChecksFunction or ChecksArguments.
func (c *Client) Checks() string { return c.checks }

// History is HistoryCumulative or HistoryLatest.
func (c *Client) History() string { return c.history }

// Feedback is the feedback representation.
func (c *Client) Feedback() string { return c.feedback }

// OnExhaustion is ExhaustionFirst or ExhaustionLast.
func (c *Client) OnExhaustion() string { return c.onExhaustion }

// PreserveInitialNoCall reports whether a readable initial no call is returned without verification.
func (c *Client) PreserveInitialNoCall() bool { return c.preserveInitialNoCall }

// ExecuteUnverified reports whether Run executes a final candidate that was not accepted.
func (c *Client) ExecuteUnverified() bool { return c.executeUnverified }

// GoString uses the same sanitized description for fmt's %#v format.
func (c Client) GoString() string { return c.String() }

// String describes the client's settings. It has a value receiver so that a dereferenced
// Client is formatted through it too: fmt would otherwise print the fields, keys included.
func (c Client) String() string {
	return fmt.Sprintf("DMCall(model=%v, decision_model=%v, checks=%s, history=%s, feedback=%s, max_repairs=%d, "+
		"on_exhaustion=%s, preserve_initial_no_call=%s)", c.generator, c.decisionModel, pyStr(c.checks),
		pyStr(c.history), pyStr(c.feedback), c.maxRepairs, pyStr(c.onExhaustion), pyRepr(c.preserveInitialNoCall))
}

func callOptions(opts []CallOption) (*callConfig, error) {
	config := &callConfig{}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt.applyCall(config); err != nil {
			return nil, err
		}
	}
	return config, nil
}

// Call proposes, verifies and repairs a call. Nothing is executed.
func (c *Client) Call(ctx context.Context, messages []Message, opts ...CallOption) (*Result, error) {
	config, err := callOptions(opts)
	if err != nil {
		return nil, err
	}
	return c.solve(ctx, messages, config, false, false)
}

// Run is Call, then it executes the final call with its tool's Handler if the call was accepted.
// With WithExecuteUnverified(true) the final candidate also runs when none was accepted: the first
// one, or the last with ExhaustionLast. An unreadable candidate or a no call never runs. If a handler
// fails, Run returns the result with the outputs gathered so far and the error. Every tool must have
// a Handler before any request is sent; per-call tools keep registered handlers of the same name.
func (c *Client) Run(ctx context.Context, messages []Message, opts ...CallOption) (*Result, error) {
	config, err := callOptions(opts)
	if err != nil {
		return nil, err
	}
	unverified := c.executeUnverified
	if config.executeUnverified != nil {
		unverified = *config.executeUnverified
	}
	return c.solve(ctx, messages, config, true, unverified)
}

// Verify checks calls from any source (see NormalizeCalls) with the schema check and the decision
// model's questions; nothing is generated or executed. messages is the conversation so far,
// including earlier tool calls and tool results. It takes only WithTools.
func (c *Client) Verify(ctx context.Context, messages []Message, calls any, opts ...CallOption) (*Verdict, error) {
	config, err := callOptions(opts)
	if err != nil {
		return nil, err
	}
	if config.other != "" {
		return nil, fmt.Errorf("dmcall: Verify takes only WithTools, not %s", config.other)
	}
	tools := c.tools
	if config.tools != nil {
		tools = *config.tools
	}
	return verdict(ctx, c.decisionModel, messages, calls, tools)
}

// Verify checks calls from any source before running them; nothing is generated or executed. The
// calls pass if they pass the schema check and the decision model's function question and, unless
// there is no call, its argument question. A nil decisionModel means Jev with TYPESAFE_API_KEY and
// threshold 0.5.
func Verify(ctx context.Context, decisionModel DecisionModel, messages []Message, calls any, tools []Tool) (*Verdict, error) {
	if decisionModel == nil {
		jev, err := NewJev()
		if err != nil {
			return nil, err
		}
		decisionModel = jev
	}
	return verdict(ctx, decisionModel, messages, calls, tools)
}

// Verdict is the schema check's and the decision model's judgment of one set of calls.
type Verdict struct {
	Accepted             bool
	Calls                []Call
	FunctionProbability  *float64
	ArgumentsProbability *float64
	SchemaErrors         []SchemaError
	Feedback             *Object
	Raw                  *Judgment
}

func verdict(ctx context.Context, decisionModel DecisionModel, messages []Message, calls any, tools []Tool) (*Verdict, error) {
	schemas, _, err := toolset(tools, nil)
	if err != nil {
		return nil, err
	}
	turns, err := conversation(messages)
	if err != nil {
		return nil, err
	}
	proposed, err := NormalizeCalls(calls)
	if err != nil {
		return nil, err
	}
	local, err := CheckSchema(schemas, proposed)
	if err != nil {
		return nil, err
	}
	check, err := decisionModel.Judge(ctx, turns, schemas, proposed)
	if err != nil {
		return nil, err
	}
	accepted := local.Accepted && check.FunctionPass && (check.ArgumentsSkipped || deref(check.ArgumentsPass))
	return &Verdict{Accepted: accepted, Calls: proposed, FunctionProbability: check.FunctionProbability,
		ArgumentsProbability: check.ArgumentsProbability, SchemaErrors: local.Errors, Feedback: check.Feedback,
		Raw: check}, nil
}

// Result is the outcome of one DM-Call request.
type Result struct {
	// Calls is the final candidate: empty for a decision to make no call, nil for an unreadable
	// candidate (cut off, filtered or refused).
	Calls []Call
	// Accepted says whether the final candidate passed every check; nil for an initial no call
	// returned without verification (InitialNoCallPreserved).
	Accepted *bool
	// Exhausted says no candidate was accepted within the repair limit.
	Exhausted bool
	Repairs   int
	// Text is the text the generator wrote with the final candidate ("" if none).
	Text     string
	Executed bool
	Outputs  []any
	Latency  time.Duration
	// Trace keeps every candidate, judgment and feedback record in order.
	Trace []Step
	// FinalAttempt is the index of the final candidate in Trace.
	FinalAttempt           int
	InitialNoCallPreserved bool
}

// Step is one round of the loop.
type Step struct {
	Attempt  int
	Proposal *Proposal
	// Verification is nil for an initial no call returned without verification.
	Verification *Judgment
	Schema       *SchemaCheck
	// Feedback is the record behind the generator's view of this round (before representation).
	Feedback *Object
	// Accepted is nil for an initial no call returned without verification.
	Accepted *bool
}

// Attempt summarizes one candidate.
type Attempt struct {
	Calls                []Call
	Accepted             *bool
	FunctionProbability  *float64
	ArgumentsProbability *float64
	SchemaErrors         []SchemaError
}

// Probabilities are the decision model's probabilities of yes for a candidate (nil if not asked).
type Probabilities struct {
	Function  *float64
	Arguments *float64
}

// UsageTotals are token counts summed over every generator request and every decision-model request.
type UsageTotals struct {
	Generator     Usage
	DecisionModel Usage
}

// Call is the first call of the final candidate, or nil for no call or an unreadable candidate.
func (r *Result) Call() *Call {
	if len(r.Calls) == 0 {
		return nil
	}
	call := r.Calls[0]
	return &call
}

// NoCall reports a decision to make no call.
func (r *Result) NoCall() bool { return r.Calls != nil && len(r.Calls) == 0 }

// Output is what the executed handler returned (a []any if several calls ran).
func (r *Result) Output() any {
	switch len(r.Outputs) {
	case 0:
		return nil
	case 1:
		return r.Outputs[0]
	}
	return append([]any(nil), r.Outputs...)
}

// Attempts returns each candidate with its verdict, scores and schema errors, oldest first.
func (r *Result) Attempts() []Attempt {
	out := make([]Attempt, len(r.Trace))
	for i, step := range r.Trace {
		attempt := Attempt{Calls: cloneCalls(step.Proposal.Calls), Accepted: step.Accepted, SchemaErrors: []SchemaError{}}
		if step.Verification != nil {
			attempt.FunctionProbability = step.Verification.FunctionProbability
			attempt.ArgumentsProbability = step.Verification.ArgumentsProbability
		}
		if step.Schema != nil {
			attempt.SchemaErrors = append(attempt.SchemaErrors, step.Schema.Errors...)
		}
		out[i] = attempt
	}
	return out
}

// Probabilities returns the decision model's probabilities of yes for the final candidate.
func (r *Result) Probabilities() Probabilities {
	if len(r.Trace) == 0 || r.FinalAttempt < 0 || r.FinalAttempt >= len(r.Trace) {
		return Probabilities{}
	}
	check := r.Trace[r.FinalAttempt].Verification
	if check == nil {
		return Probabilities{}
	}
	return Probabilities{Function: check.FunctionProbability, Arguments: check.ArgumentsProbability}
}

// Usage sums token counts over every generator request and every decision-model request.
func (r *Result) Usage() UsageTotals {
	totals := UsageTotals{Generator: Usage{}, DecisionModel: Usage{}}
	for _, step := range r.Trace {
		if step.Proposal != nil {
			totals.Generator.add(step.Proposal.Usage)
		}
		if step.Verification != nil {
			totals.DecisionModel.add(step.Verification.Usage)
		}
	}
	return totals
}

// String summarizes the result: Result(call=get_weather(city="Seoul"), accepted=True, repairs=0).
func (r *Result) String() string {
	var decision string
	switch {
	case r.Calls == nil:
		decision = "unreadable candidate"
	case len(r.Calls) == 0:
		decision = "no call"
	default:
		calls := make([]string, len(r.Calls))
		for i, call := range r.Calls {
			calls[i] = call.String()
		}
		decision = "call=" + strings.Join(calls, "; ")
	}
	accepted := "None"
	if r.Accepted != nil {
		accepted = pyRepr(*r.Accepted)
	}
	parts := []string{decision, "accepted=" + accepted, fmt.Sprintf("repairs=%d", r.Repairs)}
	if r.Executed {
		parts = append(parts, "output="+pyRepr(r.Output()))
	}
	return "Result(" + strings.Join(parts, ", ") + ")"
}

// The loop ---------------------------------------------------------------------------------------------

// accept is A_t (Eq. 2) for the configured questions, and the feedback record the generator gets.
func (c *Client) accept(local *SchemaCheck, check *Judgment) (bool, *Object, error) {
	arguments := check.ArgumentsSkipped || deref(check.ArgumentsPass)
	switch c.checks {
	case ChecksFunction:
		feedback, err := SingleCheck(check.Feedback, "function")
		return local.Accepted && check.FunctionPass, feedback, err
	case ChecksArguments:
		feedback, err := SingleCheck(check.Feedback, "arguments")
		return local.Accepted && arguments, feedback, err
	}
	return local.Accepted && check.FunctionPass && arguments, check.Feedback.Clone(), nil
}

// readableNoCall is a completely parsed decision to make no call. Unreadable output is not one.
func readableNoCall(proposal *Proposal) bool {
	return proposal.Calls != nil && len(proposal.Calls) == 0 && proposal.Incomplete == "" &&
		len(proposal.ParseErrors) == 0
}

func boolPointer(value bool) *bool { return &value }

func (c *Client) solve(ctx context.Context, messages []Message, config *callConfig, execute, unverified bool) (*Result, error) {
	limit := c.maxRepairs
	if config.maxRepairs != nil {
		if err := checkRepairs(*config.maxRepairs); err != nil {
			return nil, err
		}
		limit = *config.maxRepairs
	}
	rule := c.onExhaustion
	if config.onExhaustion != nil {
		if err := choice("on_exhaustion", *config.onExhaustion, ExhaustionFirst, ExhaustionLast); err != nil {
			return nil, err
		}
		rule = *config.onExhaustion
	}
	var schemas []Tool
	var registry map[string]Handler
	var err error
	if config.tools == nil {
		handlers := map[string]Handler{}
		for name, handler := range c.handlers {
			handlers[name] = handler
		}
		for name, handler := range config.handlers {
			handlers[name] = handler
		}
		schemas, registry, err = toolset(c.tools, handlers)
	} else {
		schemas, registry, err = toolset(*config.tools, config.handlers)
		if err == nil {
			_, known, knownErr := toolset(c.tools, c.handlers)
			if knownErr != nil {
				return nil, knownErr
			}
			for _, tool := range schemas {
				if registry[tool.Name] == nil && known[tool.Name] != nil {
					registry[tool.Name] = known[tool.Name]
				}
			}
		}
	}
	if err != nil {
		return nil, err
	}
	if execute {
		for _, tool := range schemas {
			if registry[tool.Name] == nil {
				return nil, fmt.Errorf("dmcall: no handler for tool %s; pass WithHandlers or use Call", pyStr(tool.Name))
			}
		}
	}
	turns, err := generatorMessages(messages)
	if err != nil {
		return nil, err
	}
	compiled, err := compileTools(schemas)
	if err != nil {
		return nil, err
	}
	example := Example{Messages: turns, Tools: schemas}
	started := time.Now()
	var history []HistoryEntry
	var trace []Step
	accepted := false
	for attempt := 0; attempt <= limit; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		shown := make([]HistoryEntry, len(history))
		for i, item := range history {
			shown[i] = HistoryEntry{Round: i + 1, Calls: item.Calls, Feedback: item.Feedback}
		}
		if c.history == HistoryLatest && len(history) > 0 {
			// Only the latest rejection, with its real round number.
			latest := history[len(history)-1]
			shown = []HistoryEntry{{Round: attempt, Calls: latest.Calls, Feedback: latest.Feedback}}
		}
		represented, err := RepresentHistory(shown, c.feedback)
		if err != nil {
			return nil, err
		}
		proposal, err := c.generator.Generate(ctx, example, represented)
		if err != nil {
			return nil, err
		}
		if proposal == nil {
			return nil, errors.New("dmcall: the generator returned no proposal")
		}
		local, err := compiled.check(callsValue(proposal.Calls))
		if err != nil {
			return nil, err
		}
		if attempt == 0 && c.preserveInitialNoCall && readableNoCall(proposal) && local.Accepted {
			// A readable initial no call is returned without verification or repair.
			trace = append(trace, Step{Attempt: 0, Proposal: proposal, Schema: local, Feedback: NewObject()})
			return &Result{Calls: []Call{}, Exhausted: false, Repairs: 0, Text: proposal.Content,
				Latency: time.Since(started), Trace: trace, FinalAttempt: 0, InitialNoCallPreserved: true}, nil
		}
		check, err := c.decisionModel.Judge(ctx, example.Messages, schemas, proposal.Calls)
		if err != nil {
			return nil, err
		}
		if check == nil {
			return nil, errors.New("dmcall: the decision model returned no judgment")
		}
		pass, feedback, err := c.accept(local, check)
		if err != nil {
			return nil, err
		}
		accepted = pass
		trace = append(trace, Step{Attempt: attempt, Proposal: proposal, Verification: check, Schema: local,
			Feedback: feedback.Clone(), Accepted: boolPointer(pass)})
		if accepted {
			break
		}
		history = append(history, HistoryEntry{Calls: cloneCalls(proposal.Calls), Feedback: feedback.Clone()})
	}

	exhausted := !accepted
	final := len(trace) - 1
	if exhausted && rule == ExhaustionFirst {
		final = 0
	}
	result := &Result{Calls: cloneCalls(trace[final].Proposal.Calls), Accepted: boolPointer(accepted),
		Exhausted: exhausted, Repairs: len(trace) - 1, Text: trace[final].Proposal.Content, Trace: trace,
		FinalAttempt: final}
	var runErr error
	if execute && (accepted || unverified) {
		result.Executed, runErr = runCalls(ctx, result.Calls, registry, &result.Outputs)
	}
	result.Latency = time.Since(started)
	return result, runErr
}

// runCalls runs the final calls with their handlers. An unreadable candidate never runs.
func runCalls(ctx context.Context, calls []Call, registry map[string]Handler, outputs *[]any) (bool, error) {
	if calls == nil {
		return false, nil
	}
	for _, call := range calls {
		if call.Name == "" || call.Arguments == nil {
			return false, nil
		}
	}
	for _, call := range calls {
		if registry[call.Name] == nil {
			return false, fmt.Errorf("dmcall: no handler for tool %s; give the tool a Handler or pass WithHandlers",
				pyStr(call.Name))
		}
	}
	for _, call := range calls {
		output, err := registry[call.Name](ctx, call.Arguments.Clone())
		if err != nil {
			return false, fmt.Errorf("dmcall: tool %s failed: %w", pyStr(call.Name), err)
		}
		*outputs = append(*outputs, output)
	}
	return len(calls) > 0, nil
}
