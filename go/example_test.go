package dmcall_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	dmcall "github.com/gogoymh/decision-model-guided-function-calling/go"
)

// serve answers every request with the same JSON body, so the example runs offline.
type serve string

func (s serve) RoundTrip(r *http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}},
		Body: io.NopCloser(strings.NewReader(string(s))), Request: r}, nil
}

type weatherArgs struct {
	City string `json:"city"`
}

func Example() {
	weather := dmcall.Tool{
		Name:        "get_weather",
		Description: "Current weather for a city.",
		Parameters: dmcall.MustParseJSON(`{"type": "object", "properties": {"city": {"type": "string"}},
			"required": ["city"]}`),
		Handler: dmcall.TypedHandler(func(ctx context.Context, in weatherArgs) (map[string]any, error) {
			return map[string]any{"city": in.City, "temp": 12}, nil
		}),
	}
	// Recorded replies stand in for OpenAI and Jev.
	openai := &http.Client{Transport: serve(`{"model": "gpt-5-nano", "choices": [{"finish_reason": "tool_calls",
		"message": {"role": "assistant", "content": null, "tool_calls": [{"id": "c0", "type": "function",
		"function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}]}}]}`)}
	jev := &http.Client{Transport: serve(`{"model": "jev-1.13.0", "answers": {
		"function_correct": {"type": "noul", "noul": 0.97}, "arguments_correct": {"type": "noul", "noul": 0.94}},
		"usage": {"input_tokens": 400, "output_tokens": 0}}`)}

	generator, _ := dmcall.NewOpenAI("gpt-5-nano", dmcall.WithAPIKey("sk-example"), dmcall.WithHTTPClient(openai))
	decision, _ := dmcall.NewJev(dmcall.WithAPIKey("jev-example"), dmcall.WithHTTPClient(jev))
	client, _ := dmcall.New(generator, dmcall.WithDecisionModel(decision), dmcall.WithTools(weather))

	result, err := client.Run(context.Background(), dmcall.Prompt("Do I need a jacket in Seoul tonight?"))
	if err != nil {
		fmt.Println(err)
		return
	}
	probabilities := result.Probabilities()
	fmt.Println(result)
	fmt.Println(*probabilities.Function, *probabilities.Arguments)
	// Output:
	// Result(call=get_weather(city="Seoul"), accepted=True, repairs=0, output={'city': 'Seoul', 'temp': 12})
	// 0.97 0.94
}
