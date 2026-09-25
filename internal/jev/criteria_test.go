package jev

import (
	"strings"
	"testing"

	"github.com/sdehm/jev-classify-test/internal/benchmark"
)

func TestEnrichCriteriaUsesOnlyTrainingExamples(t *testing.T) {
	criteria := map[string]string{"label": "Description."}
	records := []benchmark.DatasetRecord{
		{ID: "train", Text: "training text", Split: benchmark.SplitTrain, GoldLabel: "label"},
		{ID: "validation", Text: "validation text", Split: benchmark.SplitValidation, GoldLabel: "label"},
		{ID: "test", Text: "test text", Split: benchmark.SplitTest, GoldLabel: "label"},
	}
	enriched, err := EnrichCriteriaWithTrainingExamples(records, criteria, 1, "seed")
	if err != nil {
		t.Fatalf("EnrichCriteriaWithTrainingExamples() error = %v", err)
	}
	if !strings.Contains(enriched["label"], "training text") {
		t.Fatalf("criteria does not contain training text: %q", enriched["label"])
	}
	if strings.Contains(enriched["label"], "validation text") || strings.Contains(enriched["label"], "test text") {
		t.Fatalf("criteria contains held-out text: %q", enriched["label"])
	}
}

func TestCriteriaHashIsStable(t *testing.T) {
	first, err := CriteriaHash(map[string]string{"a": "one", "b": "two"})
	if err != nil {
		t.Fatalf("CriteriaHash() error = %v", err)
	}
	second, err := CriteriaHash(map[string]string{"b": "two", "a": "one"})
	if err != nil {
		t.Fatalf("CriteriaHash() error = %v", err)
	}
	if first != second {
		t.Fatalf("criteria hashes differ: %q != %q", first, second)
	}
}
