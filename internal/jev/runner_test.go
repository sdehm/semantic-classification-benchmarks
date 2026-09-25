package jev

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/sdehm/semantic-classification-benchmarks/internal/benchmark"
)

type fakeEvaluator struct {
	calls int
}

func (fake *fakeEvaluator) Evaluate(_ context.Context, request SystemOneRequest) (SystemOneResponse, error) {
	fake.calls++
	return SystemOneResponse{
		Model: "jev-1.13.0",
		Answers: map[string]Answer{
			"label": {
				Type:          "choice",
				Choice:        "card_arrival",
				Probabilities: map[string]float64{"card_arrival": 0.9, "failed_transfer": 0.1},
				Confidence:    float64Pointer(0.8),
			},
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 1},
	}, nil
}

func TestRunChoiceResumesFromCheckpoint(t *testing.T) {
	fake := &fakeEvaluator{}
	records := validationRecords()
	path := t.TempDir()
	config := ChoiceRunConfig{
		Evaluator:                fake,
		Records:                  records,
		Criteria:                 criteria(),
		Model:                    "jev-1.13.0",
		Split:                    benchmark.SplitValidation,
		Limit:                    1,
		SampleSeed:               "jev-sample-v1",
		MaxCostUSD:               1,
		InputPricePerMillion:     1,
		MaxInputTokensPerRequest: 100,
		CheckpointPath:           path + "/checkpoint.jsonl",
		OutputPath:               path + "/predictions.parquet",
	}
	report, err := RunChoice(context.Background(), config)
	if err != nil {
		t.Fatalf("RunChoice() first run error = %v", err)
	}
	if report.CompletedPredictions != 1 || fake.calls != 1 {
		t.Fatalf("first report = %#v, calls = %d", report, fake.calls)
	}

	config.Limit = 2
	report, err = RunChoice(context.Background(), config)
	if err != nil {
		t.Fatalf("RunChoice() resume error = %v", err)
	}
	if report.CompletedPredictions != 2 || fake.calls != 2 {
		t.Fatalf("resumed report = %#v, calls = %d", report, fake.calls)
	}
	predictions, err := benchmark.ReadPredictionRecords(config.OutputPath)
	if err != nil {
		t.Fatalf("ReadPredictionRecords() error = %v", err)
	}
	if len(predictions) != 2 {
		t.Fatalf("prediction count = %d, want 2", len(predictions))
	}
}

func TestRunChoiceRejectsChangedCheckpointProvenance(t *testing.T) {
	fake := &fakeEvaluator{}
	config := ChoiceRunConfig{
		Evaluator: fake, Records: validationRecords(), Criteria: criteria(),
		Model: "jev-1.13.0", Split: benchmark.SplitValidation, Limit: 2,
		SampleSeed: "jev-sample-v1", MaxCostUSD: 1, InputPricePerMillion: 1,
		MaxInputTokensPerRequest: 100, CheckpointPath: t.TempDir() + "/checkpoint.jsonl",
		OutputPath: t.TempDir() + "/predictions.parquet",
	}
	if _, err := RunChoice(context.Background(), config); err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*ChoiceRunConfig){
		"criteria": func(c *ChoiceRunConfig) { c.Criteria["card_arrival"] = "Different meaning." },
		"model":    func(c *ChoiceRunConfig) { c.Model = "jev-other" },
		"seed":     func(c *ChoiceRunConfig) { c.SampleSeed = "different" },
		"text":     func(c *ChoiceRunConfig) { c.Records[0].Text = "Changed request text" },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			changed := config
			changed.Evaluator = nil
			changed.Criteria = criteria()
			changed.Records = validationRecords()
			change(&changed)
			if _, err := RunChoice(context.Background(), changed); err == nil || !strings.Contains(err.Error(), "provenance mismatch") {
				t.Fatalf("RunChoice() error = %v, want provenance mismatch", err)
			}
		})
	}
	if fake.calls != 2 {
		t.Fatalf("unexpected API calls: %d", fake.calls)
	}
}

func TestRunChoiceLegacyCheckpointRequiresExplicitCompleteReplay(t *testing.T) {
	fake := &fakeEvaluator{}
	config := ChoiceRunConfig{
		Evaluator: fake, Records: validationRecords(), Criteria: criteria(),
		Model: "jev-1.13.0", Split: benchmark.SplitValidation, Limit: 1,
		SampleSeed: "jev-sample-v1", MaxCostUSD: 1, InputPricePerMillion: 1,
		MaxInputTokensPerRequest: 100, CheckpointPath: t.TempDir() + "/checkpoint.jsonl",
		OutputPath: t.TempDir() + "/replay.parquet",
	}
	if _, err := RunChoice(context.Background(), config); err != nil {
		t.Fatal(err)
	}
	removeCheckpointProvenance(t, config.CheckpointPath)
	config.Evaluator = nil
	if _, err := RunChoice(context.Background(), config); err == nil || !strings.Contains(err.Error(), "no provenance") {
		t.Fatalf("legacy replay without opt-in error = %v", err)
	}
	config.AllowLegacyCheckpointReplay = true
	report, err := RunChoice(context.Background(), config)
	if err != nil || !report.LegacyCheckpoint {
		t.Fatalf("legacy replay report = %#v, error = %v", report, err)
	}
	if _, err := benchmark.ReadPredictionRecords(config.OutputPath); err != nil {
		t.Fatalf("legacy replay did not rebuild predictions: %v", err)
	}
	config.Limit = 2
	if _, err := RunChoice(context.Background(), config); err == nil || !strings.Contains(err.Error(), "incomplete") {
		t.Fatalf("partial legacy checkpoint error = %v", err)
	}
	if fake.calls != 1 {
		t.Fatalf("unexpected API calls: %d", fake.calls)
	}
}

func removeCheckpointProvenance(t *testing.T, path string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := bytes.Split(bytes.TrimSpace(data), []byte("\n"))
	var output bytes.Buffer
	for _, line := range lines {
		var entry map[string]json.RawMessage
		if err := json.Unmarshal(line, &entry); err != nil {
			t.Fatal(err)
		}
		delete(entry, "provenance_hash")
		if err := json.NewEncoder(&output).Encode(entry); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(path, output.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestRunChoiceStopsBeforeBudget(t *testing.T) {
	fake := &fakeEvaluator{}
	config := ChoiceRunConfig{
		Evaluator:                fake,
		Records:                  validationRecords(),
		Criteria:                 criteria(),
		Model:                    "jev-1.13.0",
		Split:                    benchmark.SplitValidation,
		Limit:                    2,
		SampleSeed:               "jev-sample-v1",
		MaxCostUSD:               0.000105,
		InputPricePerMillion:     1,
		MaxInputTokensPerRequest: 100,
		CheckpointPath:           t.TempDir() + "/checkpoint.jsonl",
		OutputPath:               t.TempDir() + "/predictions.parquet",
	}
	report, err := RunChoice(context.Background(), config)
	if err == nil {
		t.Fatal("RunChoice() returned nil error")
	}
	if report.CompletedPredictions != 1 || fake.calls != 1 {
		t.Fatalf("report = %#v, calls = %d", report, fake.calls)
	}
}

func TestRunChoiceDryRunNeedsNoEvaluator(t *testing.T) {
	config := ChoiceRunConfig{
		Records:                  validationRecords(),
		Criteria:                 criteria(),
		Model:                    "jev-1.13.0",
		Split:                    benchmark.SplitValidation,
		Limit:                    2,
		SampleSeed:               "jev-sample-v1",
		MaxCostUSD:               1,
		InputPricePerMillion:     1,
		MaxInputTokensPerRequest: 100,
		CheckpointPath:           t.TempDir() + "/checkpoint.jsonl",
		OutputPath:               t.TempDir() + "/predictions.parquet",
		DryRun:                   true,
	}
	report, err := RunChoice(context.Background(), config)
	if err != nil {
		t.Fatalf("RunChoice() dry run error = %v", err)
	}
	if report.PlannedRequests != 2 || report.CompletedPredictions != 0 {
		t.Fatalf("report = %#v", report)
	}
}

func TestSelectRecordsIsDeterministicAndStratified(t *testing.T) {
	records := []benchmark.DatasetRecord{
		{ID: "a-1", Text: "a", Split: benchmark.SplitValidation, GoldLabel: "a"},
		{ID: "a-2", Text: "a", Split: benchmark.SplitValidation, GoldLabel: "a"},
		{ID: "b-1", Text: "b", Split: benchmark.SplitValidation, GoldLabel: "b"},
		{ID: "b-2", Text: "b", Split: benchmark.SplitValidation, GoldLabel: "b"},
	}
	first := selectRecords(records, 2, "seed")
	second := selectRecords(records, 2, "seed")
	if len(first) != 2 || first[0] != second[0] || first[1] != second[1] {
		t.Fatalf("selection is not deterministic: %#v and %#v", first, second)
	}
	if first[0].GoldLabel == first[1].GoldLabel {
		t.Fatalf("selection is not stratified: %#v", first)
	}
}

func validationRecords() []benchmark.DatasetRecord {
	return []benchmark.DatasetRecord{
		{ID: "one", Text: "My card is late", Split: benchmark.SplitValidation, GoldLabel: "card_arrival"},
		{ID: "two", Text: "My card is late", Split: benchmark.SplitValidation, GoldLabel: "card_arrival"},
	}
}

func criteria() map[string]string {
	return map[string]string{
		"card_arrival":    "Card delivery.",
		"failed_transfer": "Failed transfer.",
	}
}

func float64Pointer(value float64) *float64 {
	return &value
}
