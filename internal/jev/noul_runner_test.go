package jev

import (
	"context"
	"strings"
	"testing"

	"github.com/sdehm/jev-classify-test/internal/benchmark"
)

type fakeNoulEvaluator struct {
	calls int
}

func (fake *fakeNoulEvaluator) Evaluate(_ context.Context, request SystemOneRequest) (SystemOneResponse, error) {
	fake.calls++
	comment := request.State.(map[string]string)["comment"]
	probability := 0.1
	if comment == "joyful" {
		probability = 0.8
	}
	return SystemOneResponse{
		Model: "jev-1.13.0",
		Answers: map[string]Answer{
			"joy": {Type: "noul", Noul: &probability},
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 1},
	}, nil
}

type genericNoulEvaluator struct {
	request SystemOneRequest
}

func (fake *genericNoulEvaluator) Evaluate(_ context.Context, request SystemOneRequest) (SystemOneResponse, error) {
	fake.request = request
	probability := 0.4
	return SystemOneResponse{
		Model: "jev-1.13.0",
		Answers: map[string]Answer{
			"100145: law": {Type: "noul", Noul: &probability},
		},
		Usage: Usage{InputTokens: 10, OutputTokens: 1},
	}, nil
}

func TestRunNoulCheckpointsAndReplaysThreshold(t *testing.T) {
	fake := &fakeNoulEvaluator{}
	path := t.TempDir()
	config := NoulRunConfig{
		Evaluator:                fake,
		Records:                  noulValidationRecords(),
		Criteria:                 map[string]NoulCriteria{"joy": {True: "Joy", False: "No joy"}},
		Model:                    "jev-1.13.0",
		Split:                    benchmark.SplitValidation,
		Limit:                    2,
		SampleSeed:               "noul-sample-v1",
		Threshold:                0.5,
		MaxCostUSD:               1,
		InputPricePerMillion:     1,
		MaxInputTokensPerRequest: 100,
		CheckpointPath:           path + "/checkpoint.jsonl",
		OutputPath:               path + "/predictions.parquet",
	}
	report, err := RunNoul(context.Background(), config)
	if err != nil {
		t.Fatalf("RunNoul() error = %v", err)
	}
	if report.CompletedPredictions != 2 || fake.calls != 2 {
		t.Fatalf("report = %#v, calls = %d", report, fake.calls)
	}
	assertNoulPredictions(t, config.OutputPath, map[string]string{
		"one": `["joy"]`,
		"two": `["neutral"]`,
	})

	config.Evaluator = nil
	config.Threshold = 0.9
	if _, err := RunNoul(context.Background(), config); err != nil {
		t.Fatalf("RunNoul() threshold replay error = %v", err)
	}
	if fake.calls != 2 {
		t.Fatalf("replay calls = %d, want 2", fake.calls)
	}
	assertNoulPredictions(t, config.OutputPath, map[string]string{
		"one": `["neutral"]`,
		"two": `["neutral"]`,
	})
	config.EmptyPredictionPolicy = NoulEmptyPredictionHighestProbability
	if _, err := RunNoul(context.Background(), config); err != nil {
		t.Fatalf("RunNoul() fallback replay error = %v", err)
	}
	assertNoulPredictions(t, config.OutputPath, map[string]string{
		"one": `["joy"]`,
		"two": `["joy"]`,
	})
}

func TestRunNoulRejectsChangedCheckpointProvenance(t *testing.T) {
	fake := &fakeNoulEvaluator{}
	config := NoulRunConfig{
		Evaluator: fake, Records: noulValidationRecords(),
		Criteria: map[string]NoulCriteria{"joy": {True: "Joy", False: "No joy"}},
		Model:    "jev-1.13.0", Split: benchmark.SplitValidation, Limit: 2,
		SampleSeed: "noul-sample-v1", Threshold: 0.5, MaxCostUSD: 1,
		InputPricePerMillion: 1, MaxInputTokensPerRequest: 100,
		CheckpointPath: t.TempDir() + "/checkpoint.jsonl",
		OutputPath:     t.TempDir() + "/predictions.parquet",
	}
	if _, err := RunNoul(context.Background(), config); err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*NoulRunConfig){
		"criteria":   func(c *NoulRunConfig) { c.Criteria["joy"] = NoulCriteria{True: "Different joy", False: "No joy"} },
		"model":      func(c *NoulRunConfig) { c.Model = "jev-other" },
		"seed":       func(c *NoulRunConfig) { c.SampleSeed = "different" },
		"state":      func(c *NoulRunConfig) { c.StateField = "document" },
		"question":   func(c *NoulRunConfig) { c.QuestionTemplate = "Is %s present in `comment`?" },
		"truncation": func(c *NoulRunConfig) { c.StateTruncation, c.MaxStateCharacters = NoulStateTruncationHeadTail, 100 },
		"text":       func(c *NoulRunConfig) { c.Records[0].Text = "Changed request text" },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			changed := config
			changed.Evaluator = nil
			changed.Criteria = map[string]NoulCriteria{"joy": {True: "Joy", False: "No joy"}}
			changed.Records = noulValidationRecords()
			change(&changed)
			if _, err := RunNoul(context.Background(), changed); err == nil || !strings.Contains(err.Error(), "provenance mismatch") {
				t.Fatalf("RunNoul() error = %v, want provenance mismatch", err)
			}
		})
	}
	if fake.calls != 2 {
		t.Fatalf("unexpected API calls: %d", fake.calls)
	}
}

func TestRunNoulLegacyCheckpointRequiresExplicitCompleteReplay(t *testing.T) {
	fake := &fakeNoulEvaluator{}
	config := NoulRunConfig{
		Evaluator: fake, Records: noulValidationRecords(),
		Criteria: map[string]NoulCriteria{"joy": {True: "Joy", False: "No joy"}},
		Model:    "jev-1.13.0", Split: benchmark.SplitValidation, Limit: 1,
		SampleSeed: "noul-sample-v1", Threshold: 0.5, MaxCostUSD: 1,
		InputPricePerMillion: 1, MaxInputTokensPerRequest: 100,
		CheckpointPath: t.TempDir() + "/checkpoint.jsonl",
		OutputPath:     t.TempDir() + "/predictions.parquet",
	}
	if _, err := RunNoul(context.Background(), config); err != nil {
		t.Fatal(err)
	}
	removeCheckpointProvenance(t, config.CheckpointPath)
	config.Evaluator = nil
	if _, err := RunNoul(context.Background(), config); err == nil || !strings.Contains(err.Error(), "no provenance") {
		t.Fatalf("legacy replay without opt-in error = %v", err)
	}
	config.AllowLegacyCheckpointReplay = true
	report, err := RunNoul(context.Background(), config)
	if err != nil || !report.LegacyCheckpoint {
		t.Fatalf("legacy replay report = %#v, error = %v", report, err)
	}
	config.Limit = 2
	if _, err := RunNoul(context.Background(), config); err == nil || !strings.Contains(err.Error(), "incomplete") {
		t.Fatalf("partial legacy checkpoint error = %v", err)
	}
	if fake.calls != 1 {
		t.Fatalf("unexpected API calls: %d", fake.calls)
	}
}

func TestRunNoulDryRunNeedsNoEvaluator(t *testing.T) {
	config := NoulRunConfig{
		Records:                  noulValidationRecords(),
		Criteria:                 map[string]NoulCriteria{"joy": {True: "Joy", False: "No joy"}},
		Model:                    "jev-1.13.0",
		Split:                    benchmark.SplitValidation,
		Limit:                    2,
		SampleSeed:               "noul-sample-v1",
		Threshold:                0.5,
		MaxCostUSD:               1,
		InputPricePerMillion:     1,
		MaxInputTokensPerRequest: 100,
		CheckpointPath:           t.TempDir() + "/checkpoint.jsonl",
		OutputPath:               t.TempDir() + "/predictions.parquet",
		DryRun:                   true,
	}
	report, err := RunNoul(context.Background(), config)
	if err != nil {
		t.Fatalf("RunNoul() dry run error = %v", err)
	}
	if report.PlannedRequests != 2 || report.CompletedPredictions != 0 {
		t.Fatalf("report = %#v", report)
	}
}

func TestRunNoulSupportsGenericStateAndHighestProbabilityFallback(t *testing.T) {
	fake := &genericNoulEvaluator{}
	path := t.TempDir()
	config := NoulRunConfig{
		Evaluator: fake,
		Records: []benchmark.MultiLabelDatasetRecord{
			{ID: "legal", Text: "A legal document", Split: benchmark.SplitValidation, GoldLabelsJSON: `["100145: law"]`},
		},
		Criteria: map[string]NoulCriteria{
			"100145: law": {True: "Legal rules", False: "No legal rules"},
		},
		Model:                    "jev-1.13.0",
		Split:                    benchmark.SplitValidation,
		Limit:                    1,
		SampleSeed:               "legal-sample-v1",
		StateField:               "document",
		QuestionTemplate:         "Does `document` address %s?",
		EmptyPredictionPolicy:    NoulEmptyPredictionHighestProbability,
		Threshold:                0.5,
		MaxCostUSD:               1,
		InputPricePerMillion:     1,
		MaxInputTokensPerRequest: 100,
		CheckpointPath:           path + "/checkpoint.jsonl",
		OutputPath:               path + "/predictions.parquet",
	}
	if _, err := RunNoul(context.Background(), config); err != nil {
		t.Fatalf("RunNoul() error = %v", err)
	}
	if document := fake.request.State.(map[string]string)["document"]; document != "A legal document" {
		t.Fatalf("document state = %q", document)
	}
	if got := fake.request.Questions["100145: law"].Instructions; got != "Does `document` address 100145: law?" {
		t.Fatalf("instructions = %#v", got)
	}
	assertNoulPredictions(t, config.OutputPath, map[string]string{
		"legal": `["100145: law"]`,
	})
}

func TestTruncateNoulStatePreservesHeadAndTail(t *testing.T) {
	config := NoulRunConfig{
		MaxStateCharacters: 100,
		StateTruncation:    NoulStateTruncationHeadTail,
	}
	truncated, err := truncateNoulState("beginning "+strings.Repeat("middle ", 20)+"ending", config)
	if err != nil {
		t.Fatalf("truncateNoulState() error = %v", err)
	}
	if len([]rune(truncated)) != config.MaxStateCharacters {
		t.Fatalf("truncated state length = %d, want %d", len([]rune(truncated)), config.MaxStateCharacters)
	}
	if !strings.HasPrefix(truncated, "beginning") || !strings.HasSuffix(truncated, "ending") {
		t.Fatalf("truncated state did not preserve boundaries: %q", truncated)
	}
	if !strings.Contains(truncated, omittedStateMarker) {
		t.Fatalf("truncated state did not include omission marker: %q", truncated)
	}
}

func TestSelectMultiLabelRecordsPrioritizesCoverage(t *testing.T) {
	records := []benchmark.MultiLabelDatasetRecord{
		{ID: "joy", Text: "joy", Split: benchmark.SplitValidation, GoldLabelsJSON: `["joy"]`},
		{ID: "fear", Text: "fear", Split: benchmark.SplitValidation, GoldLabelsJSON: `["fear"]`},
		{ID: "neutral", Text: "neutral", Split: benchmark.SplitValidation, GoldLabelsJSON: `["neutral"]`},
		{ID: "joy-two", Text: "joy", Split: benchmark.SplitValidation, GoldLabelsJSON: `["joy"]`},
	}
	selected := selectMultiLabelRecords(records, 3, "seed")
	if len(selected) != 3 {
		t.Fatalf("selection length = %d, want 3", len(selected))
	}
	covered := make(map[string]bool)
	for _, record := range selected {
		labels, err := benchmark.DecodeGoldLabels(record.GoldLabelsJSON)
		if err != nil {
			t.Fatal(err)
		}
		for _, label := range labels {
			covered[label] = true
		}
	}
	for _, label := range []string{"joy", "fear", "neutral"} {
		if !covered[label] {
			t.Fatalf("selection did not cover %q: %#v", label, selected)
		}
	}
}

func assertNoulPredictions(t *testing.T, path string, want map[string]string) {
	t.Helper()
	predictions, err := benchmark.ReadMultiLabelPredictionRecords(path)
	if err != nil {
		t.Fatalf("ReadMultiLabelPredictionRecords() error = %v", err)
	}
	if len(predictions) != len(want) {
		t.Fatalf("prediction count = %d, want %d", len(predictions), len(want))
	}
	for _, prediction := range predictions {
		if prediction.PredictedLabelsJSON != want[prediction.ID] {
			t.Fatalf("prediction %q labels = %q, want %q", prediction.ID, prediction.PredictedLabelsJSON, want[prediction.ID])
		}
	}
}

func noulValidationRecords() []benchmark.MultiLabelDatasetRecord {
	return []benchmark.MultiLabelDatasetRecord{
		{ID: "one", Text: "joyful", Split: benchmark.SplitValidation, GoldLabelsJSON: `["joy"]`},
		{ID: "two", Text: "plain", Split: benchmark.SplitValidation, GoldLabelsJSON: `["neutral"]`},
	}
}
