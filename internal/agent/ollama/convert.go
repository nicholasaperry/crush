package ollama

import (
	"encoding/json"
	"os"
	"strconv"

	"charm.land/fantasy"
	"github.com/ollama/ollama/api"
)

func toChatRequest(modelID string, modelNumCtx int, call fantasy.Call) api.ChatRequest {
	req := api.ChatRequest{
		Model:    modelID,
		Messages: toAPIMessages(call.Prompt),
		Tools:    toAPITools(call.Tools),
		Stream:   ptr(true),
	}

	opts := api.DefaultOptions()
	if call.MaxOutputTokens != nil {
		opts.NumPredict = int(*call.MaxOutputTokens)
	}
	if call.Temperature != nil {
		opts.Temperature = float32(*call.Temperature)
	}
	if call.TopP != nil {
		opts.TopP = float32(*call.TopP)
	}
	if call.TopK != nil {
		opts.TopK = int(*call.TopK)
	}
	if numCtx := resolveNumCtx(modelNumCtx); numCtx > 0 {
		opts.NumCtx = numCtx
	}
	if optsBytes, err := json.Marshal(opts); err == nil {
		var optsMap map[string]any
		if json.Unmarshal(optsBytes, &optsMap) == nil {
			req.Options = optsMap
		}
	}

	return req
}

func toAPIMessages(prompt fantasy.Prompt) []api.Message {
	msgs := make([]api.Message, 0, len(prompt))
	for _, m := range prompt {
		msg := api.Message{Role: string(m.Role)}
		for _, part := range m.Content {
			switch p := part.(type) {
			case fantasy.TextPart:
				msg.Content += p.Text
			case fantasy.ReasoningPart:
				msg.Thinking += p.Text
			case fantasy.ToolCallPart:
				args := parseToolArguments(p.Input)
				msg.ToolCalls = append(msg.ToolCalls, api.ToolCall{
					ID: p.ToolCallID,
					Function: api.ToolCallFunction{
						Name:      p.ToolName,
						Arguments: args,
					},
				})
			case fantasy.ToolResultPart:
				msg.Role = "tool"
				msg.ToolCallID = p.ToolCallID
				if text, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](p.Output); ok {
					msg.Content = text.Text
				}
			}
		}
		msgs = append(msgs, msg)
	}
	return msgs
}

func toAPITools(tools []fantasy.Tool) api.Tools {
	if len(tools) == 0 {
		return nil
	}
	out := make(api.Tools, 0, len(tools))
	for _, t := range tools {
		fn, ok := t.(fantasy.FunctionTool)
		if !ok {
			continue
		}
		params := schemaToToolParams(fn.InputSchema)
		out = append(out, api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        fn.Name,
				Description: fn.Description,
				Parameters:  params,
			},
		})
	}
	return out
}

func schemaToToolParams(schema map[string]any) api.ToolFunctionParameters {
	b, err := json.Marshal(schema)
	if err != nil {
		return api.ToolFunctionParameters{Type: "object"}
	}
	var p api.ToolFunctionParameters
	if err := json.Unmarshal(b, &p); err != nil {
		return api.ToolFunctionParameters{Type: "object"}
	}
	return p
}

func parseToolArguments(input string) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	if input == "" {
		return args
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(input), &m); err != nil {
		return args
	}
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}

func resolveNumCtx(modelNumCtx int) int {
	if v := os.Getenv("CRUSH_OLLAMA_NUM_CTX"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	if v := os.Getenv("OLLAMA_CONTEXT_LENGTH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	if modelNumCtx > 0 {
		return modelNumCtx
	}
	return defaultNumCtx
}

func ptr[T any](v T) *T { return &v }
