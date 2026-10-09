package scan

// Tests for the Tier-1 reproducibility work (discussion #56): the verdict
// cache, cache keying, and temperature/model resolution.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"
)

// TestMain disables the verdict cache for the whole scan test binary by default,
// so tests that call Scan (chain/fallback tests) are hermetic and never touch
// the real ~/.cache or leak state between tests. Cache tests below opt back in
// per-test with AURSCAN_CACHE_DIR + AURSCAN_NO_CACHE="".
func TestMain(m *testing.M) {
	os.Setenv("AURSCAN_NO_CACHE", "1")
	os.Exit(m.Run())
}

func TestCacheKeyIsSensitiveToEveryInput(t *testing.T) {
	base := cacheKey("instr", "prompt", "model")
	cases := map[string]string{
		"same":        cacheKey("instr", "prompt", "model"),
		"diff instr":  cacheKey("instr2", "prompt", "model"),
		"diff prompt": cacheKey("instr", "prompt2", "model"),
		"diff model":  cacheKey("instr", "prompt", "model2"),
	}
	if cases["same"] != base {
		t.Fatal("identical inputs must produce identical keys")
	}
	for name, k := range cases {
		if name == "same" {
			continue
		}
		if k == base {
			t.Errorf("%s must change the cache key but did not", name)
		}
	}
	// A separator swap must not collide (field-boundary forgery guard).
	if cacheKey("a", "b", "c") == cacheKey("ab", "", "c") {
		t.Error("field boundaries must be unforgeable via content shifting")
	}
}

func TestCacheRoundTrip(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("AURSCAN_CACHE_DIR", dir)
	t.Setenv("AURSCAN_NO_CACHE", "")

	key := cacheKey("i", "p", "claude-sonnet-4-6")
	want := Verdict{Verdict: "OK", Confidence: 91, Summary: "clean", Findings: nil}
	cacheStore(key, "claude-sonnet-4-6", want)

	got, model, ok := cacheLoad(key)
	if !ok {
		t.Fatal("expected a cache hit after store")
	}
	if got.Verdict != "OK" || got.Confidence != 91 || model != "claude-sonnet-4-6" {
		t.Fatalf("round-trip mismatch: %+v model=%q", got, model)
	}
	// The entry is a real file under the cache dir.
	if _, err := os.Stat(filepath.Join(dir, key+".json")); err != nil {
		t.Fatalf("cache file not written: %v", err)
	}
}

func TestCacheMissOnUnknownKey(t *testing.T) {
	t.Setenv("AURSCAN_CACHE_DIR", t.TempDir())
	if _, _, ok := cacheLoad(cacheKey("x", "y", "z")); ok {
		t.Fatal("unknown key must miss")
	}
}

func TestCacheDisabledSkipsReadAndWrite(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("AURSCAN_CACHE_DIR", dir)
	t.Setenv("AURSCAN_NO_CACHE", "1")

	key := cacheKey("i", "p", "m")
	cacheStore(key, "m", Verdict{Verdict: "OK"})
	if _, err := os.Stat(filepath.Join(dir, key+".json")); !os.IsNotExist(err) {
		t.Fatal("disabled cache must not write")
	}
	if _, _, ok := cacheLoad(key); ok {
		t.Fatal("disabled cache must not read")
	}
}

func TestCacheVersionMismatchIgnored(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("AURSCAN_CACHE_DIR", dir)
	t.Setenv("AURSCAN_NO_CACHE", "")

	key := cacheKey("i", "p", "m")
	// Hand-write an entry with an incompatible schema version.
	data, _ := json.Marshal(cacheEntry{Version: "v0-old", Model: "m", Verdict: Verdict{Verdict: "OK"}})
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, key+".json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, ok := cacheLoad(key); ok {
		t.Fatal("entry from an incompatible cacheVersion must be ignored")
	}
}

func TestCacheTTLExpiry(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("AURSCAN_CACHE_DIR", dir)
	t.Setenv("AURSCAN_NO_CACHE", "")
	t.Setenv("AURSCAN_CACHE_TTL", "1") // 1 day

	key := cacheKey("i", "p", "m")
	// Write an entry timestamped two days in the past (beyond the 1-day TTL).
	old := cacheEntry{
		Version: cacheVersion,
		Model:   "m",
		Time:    time.Now().AddDate(0, 0, -2),
		Verdict: Verdict{Verdict: "OK"},
	}
	data, _ := json.Marshal(old)
	os.MkdirAll(dir, 0o700)
	os.WriteFile(filepath.Join(dir, key+".json"), data, 0o600)

	if _, _, ok := cacheLoad(key); ok {
		t.Fatal("entry older than the TTL must be treated as a miss")
	}

	// TTL=0 disables expiry: the same old entry now hits.
	t.Setenv("AURSCAN_CACHE_TTL", "0")
	if _, _, ok := cacheLoad(key); !ok {
		t.Fatal("TTL=0 must disable expiry so the old entry hits")
	}
}

func TestResolveTemperatureDefaultsToZero(t *testing.T) {
	t.Setenv("AURSCAN_TEMPERATURE", "")
	t.Setenv("AURSCAN_OPENAI_TEMPERATURE", "")
	if got := resolveTemperature(Backend{Kind: "api"}); got != 0 {
		t.Fatalf("default temperature = %v, want 0 (reproducible auditing)", got)
	}
}

func TestResolveTemperatureOverrides(t *testing.T) {
	// Backend-agnostic override.
	t.Setenv("AURSCAN_TEMPERATURE", "0.7")
	t.Setenv("AURSCAN_OPENAI_TEMPERATURE", "")
	if got := resolveTemperature(Backend{Kind: "openai"}); got != 0.7 {
		t.Fatalf("AURSCAN_TEMPERATURE override = %v, want 0.7", got)
	}
	// Per-backend value wins over env.
	f := 1.0
	if got := resolveTemperature(Backend{Kind: "openai", Temperature: &f}); got != 1.0 {
		t.Fatalf("per-backend temperature = %v, want 1.0", got)
	}
}

func TestModelIDResolution(t *testing.T) {
	t.Setenv("AURSCAN_MODEL", "")
	if got := (Backend{Kind: "api"}).ModelID(); got != "claude-sonnet-4-6" {
		t.Fatalf("api ModelID = %q, want the default model", got)
	}
	if got := (Backend{Kind: "api", Model: "claude-opus-4-8"}).ModelID(); got != "claude-opus-4-8" {
		t.Fatalf("api ModelID with explicit model = %q", got)
	}
	t.Setenv("AURSCAN_OPENAI_MODEL", "gemma-3")
	if got := (Backend{Kind: "openai"}).ModelID(); got != "gemma-3" {
		t.Fatalf("openai ModelID from env = %q, want gemma-3", got)
	}
	t.Setenv("AURSCAN_MODEL", "api-only")
	t.Setenv("AURSCAN_CLAUDE_MODEL", "opus")
	if got := (Backend{Kind: "claude"}).ModelID(); got != "opus" {
		t.Fatalf("claude ModelID from env = %q, want opus (not AURSCAN_MODEL)", got)
	}
	if got := (Backend{Kind: "cmd", Cmd: "/usr/bin/audit"}).ModelID(); got != "/usr/bin/audit" {
		t.Fatalf("cmd ModelID = %q, want the command path", got)
	}
}

// TestScanCacheReplaysWithoutCallingBackend is the crux of #56: a second scan of
// identical input must return the identical verdict WITHOUT a second backend
// call, so a re-run can never flip the verdict.
func TestScanCacheReplaysWithoutCallingBackend(t *testing.T) {
	isolateEnv(t)
	t.Setenv("AURSCAN_NO_CACHE", "") // re-enable cache for this test
	t.Setenv("AURSCAN_CACHE_DIR", t.TempDir())
	CacheBypass = false
	t.Cleanup(func() { CacheBypass = false })

	url, hits := startStub(t, 200, openAIEnvelope(`{"verdict":"OK","confidence":88,"summary":"clean"}`))
	t.Setenv("AURSCAN_BACKEND", "openai")
	t.Setenv("AURSCAN_OPENAI_URL", url)
	t.Setenv("AURSCAN_OPENAI_MODEL", "test-model")

	first := Scan("pkg", testFiles, Signals{})
	if first.V.Verdict != "OK" || first.Cached {
		t.Fatalf("first scan: verdict=%q cached=%v, want OK/false", first.V.Verdict, first.Cached)
	}
	if n := atomic.LoadInt32(hits); n != 1 {
		t.Fatalf("first scan hit backend %d time(s), want 1", n)
	}

	second := Scan("pkg", testFiles, Signals{})
	if second.V.Verdict != "OK" || !second.Cached {
		t.Fatalf("second scan: verdict=%q cached=%v, want OK/true", second.V.Verdict, second.Cached)
	}
	if n := atomic.LoadInt32(hits); n != 1 {
		t.Fatalf("second scan re-hit backend; total calls=%d, want 1 (cache must replay)", n)
	}
	if second.Model != "test-model" {
		t.Fatalf("cached model = %q, want test-model", second.Model)
	}
}

// TestScanRefreshBypassesCacheRead verifies --refresh (CacheBypass) forces a
// fresh backend call and refreshes the stored entry.
func TestScanRefreshBypassesCacheRead(t *testing.T) {
	isolateEnv(t)
	t.Setenv("AURSCAN_NO_CACHE", "")
	t.Setenv("AURSCAN_CACHE_DIR", t.TempDir())

	url, hits := startStub(t, 200, openAIEnvelope(`{"verdict":"OK","confidence":90}`))
	t.Setenv("AURSCAN_BACKEND", "openai")
	t.Setenv("AURSCAN_OPENAI_URL", url)
	t.Setenv("AURSCAN_OPENAI_MODEL", "test-model")

	CacheBypass = false
	Scan("pkg", testFiles, Signals{}) // populate cache

	CacheBypass = true
	t.Cleanup(func() { CacheBypass = false })
	res := Scan("pkg", testFiles, Signals{})
	if res.Cached {
		t.Fatal("--refresh must not serve a cached verdict")
	}
	if n := atomic.LoadInt32(hits); n != 2 {
		t.Fatalf("backend calls=%d, want 2 (populate + forced refresh)", n)
	}
}
