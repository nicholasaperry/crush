package ollama

import (
	"log/slog"
	"os"
	"path/filepath"

	"github.com/ollama/ollama/ml"
)

// InitLlamaServerPath points Ollama's runtime at a local llama-server build.
func InitLlamaServerPath() {
	if dir := existingLlamaServerDir(); dir != "" {
		ml.LibOllamaPath = dir
		slog.Info("using llama-server directory", "path", dir)
	}
	InitROCmBackend()
}

// InitROCmBackend auto-selects the ROCm backend when HIP libraries are present.
func InitROCmBackend() {
	if os.Getenv("OLLAMA_LLM_LIBRARY") != "" {
		return
	}
	dir := existingLlamaServerDir()
	if dir == "" {
		return
	}
	hipLib := filepath.Join(dir, "rocm_v7_2", "libggml-hip.so")
	if _, err := os.Stat(hipLib); err != nil {
		return
	}
	_ = os.Setenv("OLLAMA_LLM_LIBRARY", "rocm_v7_2")
	slog.Info("auto-selected GPU backend", "library", "rocm_v7_2")
}

func existingLlamaServerDir() string {
	if custom := os.Getenv("CRUSH_OLLAMA_LLM_LIBRARY"); custom != "" {
		if ok, abs := hasLlamaServer(custom); ok {
			return abs
		}
	}

	exe, err := os.Executable()
	if err == nil {
		exe, _ = filepath.EvalSymlinks(exe)
		exeDir := filepath.Dir(exe)
		for _, rel := range []string{
			filepath.Join("..", "frankenstein", "build", "lib", "ollama"),
			filepath.Join("build", "lib", "ollama"),
		} {
			if ok, abs := hasLlamaServer(filepath.Join(exeDir, rel)); ok {
				return abs
			}
		}
	}

	if wd, err := os.Getwd(); err == nil {
		for _, rel := range []string{
			filepath.Join("..", "frankenstein", "build", "lib", "ollama"),
			filepath.Join("frankenstein", "build", "lib", "ollama"),
			filepath.Join("build", "lib", "ollama"),
		} {
			if ok, abs := hasLlamaServer(filepath.Join(wd, rel)); ok {
				return abs
			}
		}
	}
	return ""
}

func hasLlamaServer(dir string) (bool, string) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return false, ""
	}
	if _, err := os.Stat(filepath.Join(abs, "llama-server")); err != nil {
		return false, ""
	}
	return true, abs
}
