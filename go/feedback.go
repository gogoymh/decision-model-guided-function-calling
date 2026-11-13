package dmcall

import (
	"errors"
	"math"
	"math/big"
	"strings"
)

// Feedback representations: how the decision model's scores reach the generator (Φ in the paper).
//
// Only the generator's copy of earlier feedback changes. Acceptance always uses the numeric
// probabilities, and Result.Trace keeps them unchanged.
//
//   - FeedbackHybrid (the default): each probability keeps its value and threshold and gains a
//     verbal label, Φ(p) = (p, ℓ(p)).
//   - FeedbackNumeric: probabilities and thresholds only, Φ(p) = p.
//   - FeedbackText: labels only, Φ(p) = ℓ(p); probabilities and thresholds are removed.
//
// ℓ uses five bands of equal width, lower bound inclusive; the last band includes 1. A question
// that was not asked (the argument question on a no call) keeps a null score and a null label.

// Feedback representations.
const (
	FeedbackHybrid  = "hybrid"
	FeedbackNumeric = "numeric"
	FeedbackText    = "text"
)

// Labels are the verbal bands of ℓ, lowest first.
var Labels = [5]string{"very likely incorrect", "likely incorrect", "uncertain", "likely correct",
	"very likely correct"}

// CutPoints are the lower bounds of the second to fifth band.
var CutPoints = [4]float64{0.2, 0.4, 0.6, 0.8}

// Where a probability's label goes. Decision-model records use probability_yes.
var labelFields = map[string]string{
	"probability_yes":       "feedback_label",
	"function_probability":  "function_label",
	"arguments_probability": "arguments_label",
	"p_f":                   "function_label",
	"p_a":                   "arguments_label",
}

var thresholdFields = map[string]bool{"threshold": true, "function_threshold": true, "arguments_threshold": true}

// Names of the two decision-model checks in a feedback record.
var checkRecords = map[string]string{"function": "function_selection", "arguments": "arguments"}

var errProbability = errors.New("dmcall: feedback probabilities must be finite numbers between 0 and 1, or null")

// Label is ℓ(p), the verbal band of a probability.
func Label(p float64) (string, error) {
	if math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
		return "", errProbability
	}
	band := 0
	for _, cut := range CutPoints {
		if p >= cut {
			band++
		}
	}
	return Labels[band], nil
}

// labelOf is ℓ for a JSON value: null for a question that was not asked.
func labelOf(v any) (any, error) {
	switch x := v.(type) {
	case nil:
		return nil, nil
	case float64:
		return Label(x)
	case int64:
		return Label(float64(x))
	case *big.Int:
		f, _ := new(big.Float).SetInt(x).Float64()
		return Label(f)
	}
	return nil, errProbability
}

func checkRepresentation(representation string) error {
	switch representation {
	case FeedbackHybrid, FeedbackNumeric, FeedbackText:
		return nil
	}
	return errors.New("dmcall: feedback must be one of 'hybrid', 'numeric', 'text'")
}

// Represent returns the generator's copy of one feedback record under a representation. The input
// is not changed.
func Represent(feedback *Object, representation string) (*Object, error) {
	if err := checkRepresentation(representation); err != nil {
		return nil, err
	}
	if feedback == nil {
		return nil, errors.New("dmcall: repair feedback must be an object")
	}
	switch representation {
	case FeedbackNumeric:
		return feedback.Clone(), nil
	case FeedbackText:
		out, err := textFeedback(feedback)
		if err != nil {
			return nil, err
		}
		return out.(*Object), nil
	}
	out, err := hybridFeedback(feedback)
	if err != nil {
		return nil, err
	}
	return out.(*Object), nil
}

func hybridFeedback(v any) (any, error) {
	switch x := v.(type) {
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			value, err := hybridFeedback(item)
			if err != nil {
				return nil, err
			}
			out[i] = value
		}
		return out, nil
	case *Object:
		out := NewObject()
		for _, key := range x.keys {
			item := x.values[key]
			if key == "schema_errors" { // diagnostics about the task input, not scores
				out.Set(key, Clone(item))
			} else {
				value, err := hybridFeedback(item)
				if err != nil {
					return nil, err
				}
				out.Set(key, value)
			}
			if field, ok := labelFields[key]; ok {
				if x.Has(field) || out.Has(field) {
					return nil, errors.New("dmcall: repair feedback has conflicting added judgment fields")
				}
				label, err := labelOf(item)
				if err != nil {
					return nil, err
				}
				out.Set(field, label)
			}
		}
		return out, nil
	}
	return Clone(v), nil
}

func textFeedback(v any) (any, error) {
	switch x := v.(type) {
	case []any:
		out := make([]any, len(x))
		for i, item := range x {
			value, err := textFeedback(item)
			if err != nil {
				return nil, err
			}
			out[i] = value
		}
		return out, nil
	case *Object:
		out := NewObject()
		for _, key := range x.keys {
			item := x.values[key]
			if thresholdFields[key] {
				continue
			}
			if field, ok := labelFields[key]; ok {
				if x.Has(field) || out.Has(field) {
					return nil, errors.New("dmcall: repair feedback has conflicting judgment fields")
				}
				label, err := labelOf(item)
				if err != nil {
					return nil, err
				}
				out.Set(field, label)
				continue
			}
			if text, ok := item.(string); ok && key == "instruction" {
				out.Set(key, strings.ReplaceAll(text, "The probabilities", "The feedback labels"))
				continue
			}
			if key == "schema_errors" {
				out.Set(key, Clone(item))
				continue
			}
			value, err := textFeedback(item)
			if err != nil {
				return nil, err
			}
			out.Set(key, value)
		}
		return out, nil
	}
	return Clone(v), nil
}

// RepresentHistory returns every record of a repair history under a representation, in order.
func RepresentHistory(history []HistoryEntry, representation string) ([]HistoryEntry, error) {
	if err := checkRepresentation(representation); err != nil {
		return nil, err
	}
	out := make([]HistoryEntry, len(history))
	for i, item := range history {
		if item.Feedback == nil {
			return nil, errors.New("dmcall: each repair history record needs a feedback object")
		}
		feedback, err := Represent(item.Feedback, representation)
		if err != nil {
			return nil, err
		}
		out[i] = HistoryEntry{Round: item.Round, Calls: cloneCalls(item.Calls), Feedback: feedback}
	}
	return out, nil
}

// SingleCheck returns a feedback record without the other question, for DM-Call-Function
// (keep = "function") and DM-Call-Argument (keep = "arguments"). The source, schema errors,
// thresholds and instruction stay; only the hidden question's verdict and score are removed.
func SingleCheck(feedback *Object, keep string) (*Object, error) {
	if _, ok := checkRecords[keep]; !ok {
		return nil, errors.New("dmcall: keep must be 'function' or 'arguments'")
	}
	hidden := "function"
	if keep == "function" {
		hidden = "arguments"
	}
	out := feedback.Clone()
	if out == nil {
		out = NewObject()
	}
	if checks, ok := out.Value("checks").(*Object); ok {
		checks.Delete(checkRecords[hidden])
	}
	if failed, ok := out.Value("failed_checks").([]any); ok {
		kept := []any{}
		for _, name := range failed {
			if name != checkRecords[hidden] {
				kept = append(kept, name)
			}
		}
		out.Set("failed_checks", kept)
	}
	for _, key := range out.Keys() {
		if strings.HasPrefix(key, hidden+"_") {
			out.Delete(key)
		}
	}
	return out, nil
}
