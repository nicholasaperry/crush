package ollama

import (
	"testing"

	"github.com/charmbracelet/crush/internal/ui/util"
)

func TestParseInferenceLine(t *testing.T) {
	line := `time=2025-01-01T00:00:00.000Z level=INFO source=types.go:32 msg="inference compute" id=0 library=ROCm compute=gfx1100 name="AMD Radeon RX 7900 XTX" total="24.0 GiB" available="23.0 GiB"`
	ev, ok := ParseInferenceLine(line)
	if !ok {
		t.Fatal("expected inference compute line to parse")
	}
	if ev.Type != util.InfoTypeSuccess {
		t.Fatalf("expected success type, got %v", ev.Type)
	}
	if ev.Message == "" {
		t.Fatal("expected non-empty message")
	}
}
