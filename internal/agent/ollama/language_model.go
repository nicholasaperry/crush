package ollama

import (
	"context"
	"fmt"

	"charm.land/fantasy"
	"github.com/ollama/ollama/engine"
)

type languageModel struct {
	engine  *engine.Engine
	modelID string
	numCtx  int
}

func (m *languageModel) Provider() string { return Name }
func (m *languageModel) Model() string    { return m.modelID }

func (m *languageModel) Generate(ctx context.Context, call fantasy.Call) (*fantasy.Response, error) {
	stream, err := m.Stream(ctx, call)
	if err != nil {
		return nil, err
	}
	var text string
	var reasoning string
	var toolCalls []fantasy.ToolCallContent
	var usage fantasy.Usage
	var finish fantasy.FinishReason
	for part := range stream {
		switch part.Type {
		case fantasy.StreamPartTypeTextDelta:
			text += part.Delta
		case fantasy.StreamPartTypeReasoningDelta:
			reasoning += part.Delta
		case fantasy.StreamPartTypeToolCall:
			toolCalls = append(toolCalls, fantasy.ToolCallContent{
				ToolCallID: part.ID,
				ToolName:   part.ToolCallName,
				Input:      part.ToolCallInput,
			})
		case fantasy.StreamPartTypeFinish:
			usage = part.Usage
			finish = part.FinishReason
		case fantasy.StreamPartTypeError:
			return nil, part.Error
		}
	}
	content := fantasy.ResponseContent{}
	if reasoning != "" {
		content = append(content, fantasy.ReasoningContent{Text: reasoning})
	}
	if text != "" {
		content = append(content, fantasy.TextContent{Text: text})
	}
	for _, tc := range toolCalls {
		content = append(content, tc)
	}
	return &fantasy.Response{
		Content:      content,
		Usage:        usage,
		FinishReason: finish,
	}, nil
}

func (m *languageModel) Stream(ctx context.Context, call fantasy.Call) (fantasy.StreamResponse, error) {
	req := toChatRequest(m.modelID, m.numCtx, call)
	return func(yield func(fantasy.StreamPart) bool) {
		textActive := false
		reasonActive := false
		var usage fantasy.Usage

		err := m.engine.StreamChat(ctx, req, func(ev engine.StreamEvent) {
			if ev.ThinkingDelta != "" {
				if !reasonActive {
					reasonActive = true
					if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeReasoningStart, ID: "r0"}) {
						return
					}
				}
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeReasoningDelta,
					ID:    "r0",
					Delta: ev.ThinkingDelta,
				}) {
					return
				}
			}

			if ev.ContentDelta != "" {
				if reasonActive {
					reasonActive = false
					yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeReasoningEnd, ID: "r0"})
				}
				if !textActive {
					textActive = true
					if !yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextStart, ID: "0"}) {
						return
					}
				}
				if !yield(fantasy.StreamPart{
					Type:  fantasy.StreamPartTypeTextDelta,
					ID:    "0",
					Delta: ev.ContentDelta,
				}) {
					return
				}
			}

			for _, tc := range ev.ToolCalls {
				args := tc.Function.Arguments.String()
				if args == "" {
					args = "{}"
				}
				if textActive {
					textActive = false
					yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextEnd, ID: "0"})
				}
				if !yield(fantasy.StreamPart{
					Type:          fantasy.StreamPartTypeToolCall,
					ID:            tc.ID,
					ToolCallName:  tc.Function.Name,
					ToolCallInput: args,
				}) {
					return
				}
			}

			if ev.Metrics.EvalCount > 0 || ev.Metrics.PromptEvalCount > 0 {
				usage = fantasy.Usage{
					InputTokens:  int64(ev.Metrics.PromptEvalCount),
					OutputTokens: int64(ev.Metrics.EvalCount),
					TotalTokens:  int64(ev.Metrics.PromptEvalCount + ev.Metrics.EvalCount),
				}
			}

			if ev.Done {
				if textActive {
					yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeTextEnd, ID: "0"})
				}
				if reasonActive {
					yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeReasoningEnd, ID: "r0"})
				}
				finish := fantasy.FinishReasonStop
				if len(ev.ToolCalls) > 0 {
					finish = fantasy.FinishReasonToolCalls
				} else if ev.DoneReason == "length" {
					finish = fantasy.FinishReasonLength
				}
				yield(fantasy.StreamPart{
					Type:         fantasy.StreamPartTypeFinish,
					Usage:        usage,
					FinishReason: finish,
				})
			}
		})
		if err != nil {
			yield(fantasy.StreamPart{Type: fantasy.StreamPartTypeError, Error: err})
		}
	}, nil
}

func (m *languageModel) GenerateObject(context.Context, fantasy.ObjectCall) (*fantasy.ObjectResponse, error) {
	return nil, fmt.Errorf("ollama provider: GenerateObject not implemented")
}

func (m *languageModel) StreamObject(context.Context, fantasy.ObjectCall) (fantasy.ObjectStreamResponse, error) {
	return nil, fmt.Errorf("ollama provider: StreamObject not implemented")
}
