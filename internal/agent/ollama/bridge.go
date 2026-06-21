package ollama

import (
	"io"
	"regexp"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/crush/internal/pubsub"
	"github.com/charmbracelet/crush/internal/ui/util"
)

var (
	deviceFreeRe       = regexp.MustCompile(`using device (\S+)`)
	offloadedLayersRe  = regexp.MustCompile(`offloaded\s+(\d+)/(\d+)\s+layers`)
	modelBufferRe      = regexp.MustCompile(`model buffer size\s*=\s*([\d.]+)\s*MiB`)
	inferenceComputeRe = regexp.MustCompile(`library=(\S+)`)
)

// InferenceEvent is a parsed inference lifecycle line suitable for UI display.
type InferenceEvent struct {
	Message string
	Type    util.InfoType
	TTL     time.Duration
}

// ParseInferenceLine extracts a user-visible status message from a backend log line.
func ParseInferenceLine(line string) (InferenceEvent, bool) {
	line = strings.TrimSpace(line)
	if line == "" {
		return InferenceEvent{}, false
	}

	lower := strings.ToLower(line)

	if strings.Contains(lower, "truncating input prompt") {
		return InferenceEvent{Message: "Truncating prompt to fit context", Type: util.InfoTypeWarn, TTL: 4 * time.Second}, true
	}
	if strings.Contains(lower, "prompt eval time") || strings.Contains(lower, "prompt eval rate") {
		return InferenceEvent{Message: shortEvalLine(line), Type: util.InfoTypeInfo, TTL: 3 * time.Second}, true
	}
	if strings.Contains(lower, "eval time") && strings.Contains(lower, "tokens/s") {
		return InferenceEvent{Message: shortEvalLine(line), Type: util.InfoTypeInfo, TTL: 3 * time.Second}, true
	}

	if match := offloadedLayersRe.FindStringSubmatch(line); len(match) == 3 {
		return InferenceEvent{
			Message: "Offloaded " + match[1] + "/" + match[2] + " layers to GPU",
			Type:    util.InfoTypeSuccess,
			TTL:     4 * time.Second,
		}, true
	}
	if strings.Contains(lower, "load_tensors") || strings.Contains(lower, "loading model") {
		if match := modelBufferRe.FindStringSubmatch(line); len(match) == 2 {
			return InferenceEvent{
				Message: "Loading model (" + match[1] + " MiB)",
				Type:    util.InfoTypeInfo,
				TTL:     5 * time.Second,
			}, true
		}
		return InferenceEvent{Message: "Loading model…", Type: util.InfoTypeInfo, TTL: 5 * time.Second}, true
	}
	if match := deviceFreeRe.FindStringSubmatch(line); len(match) == 2 {
		return InferenceEvent{
			Message: "GPU " + match[1] + " ready",
			Type:    util.InfoTypeSuccess,
			TTL:     4 * time.Second,
		}, true
	}

	if strings.Contains(line, "inference compute") {
		lib := inferenceLibrary(line)
		total := inferenceField(line, "total=")
		name := inferenceField(line, "name=")
		msg := "Inference: " + lib
		if name != "" && name != "cpu" {
			msg += " · " + name
		}
		if total != "" {
			msg += " · " + total
		}
		kind := util.InfoTypeSuccess
		if strings.EqualFold(lib, "cpu") {
			kind = util.InfoTypeWarn
		}
		return InferenceEvent{Message: msg, Type: kind, TTL: 5 * time.Second}, true
	}
	if strings.Contains(lower, "library=rocm") || strings.Contains(lower, "library=cuda") || strings.Contains(lower, "library=metal") {
		lib := inferenceLibrary(line)
		total := inferenceField(line, "total=")
		msg := "GPU backend: " + lib
		if total != "" {
			msg += " · " + total
		}
		return InferenceEvent{Message: msg, Type: util.InfoTypeSuccess, TTL: 5 * time.Second}, true
	}

	if strings.Contains(lower, "waiting for llama-server") {
		return InferenceEvent{Message: "Starting model runner…", Type: util.InfoTypeInfo, TTL: 4 * time.Second}, true
	}
	if strings.Contains(lower, "llama-server started") {
		return InferenceEvent{Message: "Model ready", Type: util.InfoTypeSuccess, TTL: 3 * time.Second}, true
	}
	if strings.Contains(lower, "out of memory") || strings.Contains(lower, "hipmalloc failed") || strings.Contains(lower, "cudamalloc failed") {
		return InferenceEvent{Message: "Out of GPU memory", Type: util.InfoTypeError, TTL: 8 * time.Second}, true
	}
	if strings.Contains(lower, "error loading model") || strings.Contains(lower, "llama-server error") {
		return InferenceEvent{Message: shortErrorLine(line), Type: util.InfoTypeError, TTL: 8 * time.Second}, true
	}

	return InferenceEvent{}, false
}

// LogBridge captures backend log output, writes raw bytes to a file, and publishes
// parsed inference lifecycle events to the Crush TUI status bar.
type LogBridge struct {
	events *pubsub.Broker[tea.Msg]
	rawLog io.Writer
	buf    []byte
}

// NewLogBridge creates a bridge that forwards subprocess output to rawLog and
// publishes util.InfoMsg events for recognized inference lifecycle lines.
func NewLogBridge(events *pubsub.Broker[tea.Msg], rawLog io.Writer) *LogBridge {
	return &LogBridge{events: events, rawLog: rawLog}
}

func (b *LogBridge) Write(p []byte) (int, error) {
	if b.rawLog != nil {
		_, _ = b.rawLog.Write(p)
	}
	b.buf = append(b.buf, p...)
	for {
		i := bytesIndex(b.buf, '\n')
		if i < 0 {
			break
		}
		line := strings.TrimRight(string(b.buf[:i]), "\r")
		b.buf = b.buf[i+1:]
		b.HandleLine(line)
	}
	return len(p), nil
}

// HandleLine parses a single log line and publishes a status bar message when relevant.
func (b *LogBridge) HandleLine(line string) {
	ev, ok := ParseInferenceLine(line)
	if !ok || b.events == nil {
		return
	}
	msg := util.InfoMsg{Type: ev.Type, Msg: ev.Message, TTL: ev.TTL}
	b.events.Publish(pubsub.UpdatedEvent, msg)
}

func bytesIndex(b []byte, c byte) int {
	for i, v := range b {
		if v == c {
			return i
		}
	}
	return -1
}

func inferenceLibrary(line string) string {
	if match := inferenceComputeRe.FindStringSubmatch(line); len(match) == 2 {
		return match[1]
	}
	return "unknown"
}

func inferenceField(line, key string) string {
	idx := strings.Index(line, key)
	if idx < 0 {
		return ""
	}
	rest := line[idx+len(key):]
	if strings.HasPrefix(rest, `"`) {
		rest = rest[1:]
		if end := strings.IndexByte(rest, '"'); end >= 0 {
			return rest[:end]
		}
	}
	if end := strings.IndexAny(rest, " \t"); end >= 0 {
		return rest[:end]
	}
	return strings.TrimSpace(rest)
}

func shortEvalLine(line string) string {
	line = strings.TrimSpace(line)
	if len(line) > 72 {
		return line[:69] + "…"
	}
	return line
}

func shortErrorLine(line string) string {
	line = strings.TrimSpace(line)
	if idx := strings.Index(line, "msg="); idx >= 0 {
		line = line[idx+4:]
	}
	line = strings.Trim(line, `"`)
	if len(line) > 96 {
		return line[:93] + "…"
	}
	if line == "" {
		return "Model load failed"
	}
	return line
}
