// Package ollama implements an in-process fantasy provider backed by github.com/ollama/ollama/engine.
package ollama

import (
	"context"
	"fmt"

	"charm.land/catwalk/pkg/catwalk"
	"charm.land/fantasy"
	"github.com/ollama/ollama/engine"
)

const Name = "ollama"

// Type is the provider type string used in crush.json.
const Type = "ollama"

const defaultNumCtx = 32768

type Provider struct {
	engine *engine.Engine
	numCtx map[string]int
}

func NewProvider(eng *engine.Engine, models []catwalk.Model) *Provider {
	numCtx := make(map[string]int, len(models))
	for _, model := range models {
		if model.ContextWindow > 0 {
			numCtx[model.ID] = int(model.ContextWindow)
		}
	}
	return &Provider{engine: eng, numCtx: numCtx}
}

func (p *Provider) numCtxFor(modelID string) int {
	if n, ok := p.numCtx[modelID]; ok && n > 0 {
		return n
	}
	return defaultNumCtx
}

func (p *Provider) Name() string { return Name }

func (p *Provider) LanguageModel(_ context.Context, modelID string) (fantasy.LanguageModel, error) {
	if p.engine == nil {
		return nil, fmt.Errorf("ollama inference engine not initialized")
	}
	return &languageModel{
		engine:  p.engine,
		modelID: modelID,
		numCtx:  p.numCtxFor(modelID),
	}, nil
}
