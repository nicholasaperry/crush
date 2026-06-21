package config

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"charm.land/catwalk/pkg/catwalk"
	"github.com/ollama/ollama/engine"
)

const OllamaProviderID = "ollama"

// ApplyOllamaProvider configures crush to use the in-process Ollama engine when
// no providers are configured or CRUSH_OLLAMA=1 is set.
func ApplyOllamaProvider(ctx context.Context, store *ConfigStore, eng *engine.Engine) error {
	if eng == nil {
		return nil
	}
	if os.Getenv("CRUSH_OLLAMA") == "0" {
		return nil
	}
	if store.Config().IsConfigured() && os.Getenv("CRUSH_OLLAMA") != "1" {
		return nil
	}

	models, err := eng.ListModels(ctx)
	if err != nil {
		return fmt.Errorf("list ollama models: %w", err)
	}
	if len(models) == 0 {
		return fmt.Errorf("no local ollama models found; run `ollama pull <model>` first")
	}

	modelName := os.Getenv("CRUSH_OLLAMA_MODEL")
	if modelName == "" {
		modelName = models[0].Name
	}

	cw := models[0].Details.ContextLength
	if cw == 0 {
		cw = 32768
	}

	catModels := make([]catwalk.Model, 0, len(models))
	for _, m := range models {
		ctxLen := m.Details.ContextLength
		if ctxLen == 0 {
			ctxLen = cw
		}
		catModels = append(catModels, catwalk.Model{
			ID:               m.Name,
			Name:             m.Name,
			ContextWindow:    int64(ctxLen),
			DefaultMaxTokens: 4096,
			CanReason:        true,
		})
	}

	cfg := store.Config()
	cfg.Providers.Set(OllamaProviderID, ProviderConfig{
		ID:       OllamaProviderID,
		Name:     "Ollama",
		Type:     catwalk.Type(TypeOllama),
		FlatRate: true,
		Models:   catModels,
	})
	cfg.Models[SelectedModelTypeLarge] = SelectedModel{
		Provider: OllamaProviderID,
		Model:    modelName,
	}
	cfg.Models[SelectedModelTypeSmall] = SelectedModel{
		Provider: OllamaProviderID,
		Model:    modelName,
	}
	store.SetupAgents()
	slog.Info("configured in-process ollama provider", "model", modelName, "models", len(catModels))
	return nil
}

// TypeOllama is the provider type for in-process inference.
const TypeOllama = "ollama"
