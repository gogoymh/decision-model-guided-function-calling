// Package dmcall is DM-Call: Decision-Model-Guided Function Calling with Cumulative Verification
// Feedback.
//
// An LLM proposes a function call; a decision model checks whether the function and the arguments
// are right before anything runs; a rejected call goes back to the same LLM with every earlier
// candidate and its feedback, until a candidate passes or the repair limit is reached.
//
//	generator, _ := dmcall.NewOpenAI("gpt-5-nano")            // OPENAI_API_KEY
//	client, _ := dmcall.New(generator, dmcall.WithTools(weather)) // TYPESAFE_API_KEY for Jev
//	result, _ := client.Run(ctx, dmcall.Prompt("Is it warm in Seoul right now?"))
//	fmt.Println(result.Call(), *result.Accepted, result.Output())
//
// The package is a port of the Python reference (src/dmcall) and sends the same requests byte for
// byte where it matters: the history message the generator reads is JSON text that equals Python's
// json.dumps(value, ensure_ascii=False), which is why JSON values here keep key order and number
// kinds (see Object and ParseJSON).
package dmcall
