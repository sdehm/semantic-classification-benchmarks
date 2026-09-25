package benchmark

import (
	"math"
	"testing"
)

func TestEvaluateSingleLabelCountsMissingPredictions(t *testing.T) {
	records := []DatasetRecord{
		{ID: "1", Text: "first", Split: SplitTest, GoldLabel: "a"},
		{ID: "2", Text: "second", Split: SplitTest, GoldLabel: "b"},
		{ID: "3", Text: "third", Split: SplitTrain, GoldLabel: "a"},
	}
	predictions := []PredictionRecord{
		{ID: "1", PredictedLabel: "a", Model: "test-model", LatencyMilliseconds: 42},
	}
	report, err := EvaluateSingleLabel(records, predictions, SplitTest)
	if err != nil {
		t.Fatalf("EvaluateSingleLabel() error = %v", err)
	}
	if report.Examples != 2 || report.Scored != 1 || report.Missing != 1 || report.Coverage != 0.5 || report.Accuracy != 1 {
		t.Fatalf("report = %#v", report)
	}
	if report.MacroF1 != 1 {
		t.Fatalf("macro F1 = %v, want 1", report.MacroF1)
	}
	if report.Latency.Samples != 1 || report.Latency.P95Millis != 42 {
		t.Fatalf("latency = %#v", report.Latency)
	}
}

func TestSummarizeLatencyUsesNearestRank(t *testing.T) {
	summary := summarizeLatency([]int64{90, 10, 30, 20, 40})
	if summary.TotalMillis != 190 || summary.P50Millis != 30 || summary.P95Millis != 90 {
		t.Fatalf("summary = %#v", summary)
	}
}

func TestEvaluateSingleLabelCalculatesCalibration(t *testing.T) {
	records := []DatasetRecord{
		{ID: "1", Text: "first", Split: SplitTest, GoldLabel: "a"},
		{ID: "2", Text: "second", Split: SplitTest, GoldLabel: "b"},
	}
	predictions := []PredictionRecord{
		{
			ID: "1", PredictedLabel: "a", Model: "test-model", HasConfidence: true,
			Confidence: 0.8, LabelProbabilitiesJSON: `{"a":0.8,"b":0.2}`,
		},
		{
			ID: "2", PredictedLabel: "a", Model: "test-model", HasConfidence: true,
			Confidence: 0.6, LabelProbabilitiesJSON: `{"a":0.6,"b":0.4}`,
		},
	}
	report, err := EvaluateSingleLabel(records, predictions, SplitTest)
	if err != nil {
		t.Fatalf("EvaluateSingleLabel() error = %v", err)
	}
	if report.Calibration.ConfidenceSamples != 2 || report.Calibration.ProbabilitySamples != 2 {
		t.Fatalf("calibration samples = %#v", report.Calibration)
	}
	if math.Abs(report.Calibration.MulticlassBrier-0.2) > 1e-9 {
		t.Fatalf("Brier score = %v, want 0.2", report.Calibration.MulticlassBrier)
	}
	if math.Abs(report.Calibration.ExpectedCalibrationError-0.4) > 1e-9 {
		t.Fatalf("ECE = %v, want 0.4", report.Calibration.ExpectedCalibrationError)
	}
}
