package scan

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeClaude puts a stub `claude` on PATH that records its argv (one arg per
// line, then a MAXTOK= line with CLAUDE_CODE_MAX_OUTPUT_TOKENS) and replies
// with a minimal JSON envelope. It returns a reader for the recorded lines.
func fakeClaude(t *testing.T) func() []string {
	t.Helper()
	isolateEnv(t)
	t.Setenv("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "")
	dir := os.Getenv("PATH")
	argv := filepath.Join(dir, "argv")
	stub := "#!/bin/sh\n{ printf '%s\\n' \"$@\"; echo \"MAXTOK=$CLAUDE_CODE_MAX_OUTPUT_TOKENS\"; } > " + argv + "\ncat >/dev/null\n" +
		`echo '{"type":"result","result":"ok","total_cost_usd":0}'` + "\n"
	if err := os.WriteFile(filepath.Join(dir, "claude"), []byte(stub), 0o755); err != nil {
		t.Fatal(err)
	}
	return func() []string {
		b, err := os.ReadFile(argv)
		if err != nil {
			t.Fatal(err)
		}
		return strings.Split(strings.TrimSpace(string(b)), "\n")
	}
}

func TestClaudeCLIModel(t *testing.T) {
	for _, tc := range []struct {
		name, env, model, want string
	}{
		{"unset", "", "", ""},
		{"env", "sonnet", "", "sonnet"},
		{"config overrides env", "sonnet", "opus", "opus"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			argv := fakeClaude(t)
			t.Setenv("AURSCAN_CLAUDE_MODEL", tc.env)
			t.Setenv("AURSCAN_MODEL", "api-only") // must never reach the CLI
			if _, _, err := callClaudeCLI(context.Background(), Backend{Kind: "claude", Model: tc.model}, "INSTR", "pkg", 0); err != nil {
				t.Fatal(err)
			}
			lines := argv()
			args := lines[:len(lines)-1]
			if args[len(args)-1] != "INSTR" {
				t.Fatalf("instructions must stay the last arg, got %q", args)
			}
			joined := strings.Join(args, " ")
			if tc.want == "" {
				if strings.Contains(joined, "--model") {
					t.Fatalf("no model pinned, but got %q", args)
				}
				return
			}
			if !strings.Contains(joined, "--model "+tc.want) {
				t.Fatalf("want --model %s, got %q", tc.want, args)
			}
		})
	}
}

func TestClaudeCLIEffortAndMaxTokens(t *testing.T) {
	for _, tc := range []struct {
		name, envEffort, envMax string
		be                      Backend
		wantEffort, wantMax     string
	}{
		{"unset", "", "", Backend{}, "", ""},
		{"env", "low", "1500", Backend{}, "low", "1500"},
		{"config overrides env", "low", "1500", Backend{Effort: "high", MaxTokens: 3000}, "high", "3000"},
		{"invalid env max ignored", "", "lots", Backend{}, "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			argv := fakeClaude(t)
			t.Setenv("AURSCAN_CLAUDE_EFFORT", tc.envEffort)
			t.Setenv("AURSCAN_CLAUDE_MAX_TOKENS", tc.envMax)
			tc.be.Kind = "claude"
			if _, _, err := callClaudeCLI(context.Background(), tc.be, "INSTR", "pkg", 0); err != nil {
				t.Fatal(err)
			}
			lines := argv()
			args, maxLine := lines[:len(lines)-1], lines[len(lines)-1]
			if args[len(args)-1] != "INSTR" {
				t.Fatalf("instructions must stay the last arg, got %q", args)
			}
			joined := strings.Join(args, " ")
			if tc.wantEffort == "" && strings.Contains(joined, "--effort") {
				t.Fatalf("no effort pinned, but got %q", args)
			}
			if tc.wantEffort != "" && !strings.Contains(joined, "--effort "+tc.wantEffort) {
				t.Fatalf("want --effort %s, got %q", tc.wantEffort, args)
			}
			if maxLine != "MAXTOK="+tc.wantMax {
				t.Fatalf("CLAUDE_CODE_MAX_OUTPUT_TOKENS: got %q, want %q", maxLine, "MAXTOK="+tc.wantMax)
			}
		})
	}
}
