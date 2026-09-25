package benchmark

import (
	"math"
	"testing"
)

func TestEvaluateMultiLabel(t *testing.T) {
	records := []MultiLabelDatasetRecord{
		{ID: "1", Text: "happy", Split: SplitTest, GoldLabelsJSON: `["joy"]`},
		{ID: "2", Text: "happy and worried", Split: SplitTest, GoldLabelsJSON: `["fear","joy"]`},
		{ID: "3", Text: "neutral", Split: SplitTrain, GoldLabelsJSON: `["neutral"]`},
	}
	predictions := []MultiLabelPredictionRecord{
		{
			ID: "1", PredictedLabelsJSON: `["joy"]`,
			LabelProbabilitiesJSON: `{"fear":0.2,"joy":0.8}`, Model: "test-model",
		},
		{
			ID: "2", PredictedLabelsJSON: `["fear"]`,
			LabelProbabilitiesJSON: `{"fear":0.2,"joy":0.8}`, Model: "test-model",
		},
	}
	report, err := EvaluateMultiLabel(records, predictions, SplitTest)
	if err != nil {
		t.Fatalf("EvaluateMultiLabel() error = %v", err)
	}
	if !report.Complete || report.Examples != 2 || report.Scored != 2 {
		t.Fatalf("report coverage = %#v", report)
	}
	if report.ExactMatchRatio != 0.5 {
		t.Fatalf("exact match = %v, want 0.5", report.ExactMatchRatio)
	}
	if report.Micro.F1 != 0.8 {
		t.Fatalf("micro F1 = %v, want 0.8", report.Micro.F1)
	}
	if math.Abs(report.Macro.F1-(2.0/3.0+1.0)/3.0) > 1e-12 {
		t.Fatalf("macro F1 = %v", report.Macro.F1)
	}
	if report.Calibration.ProbabilitySamples != 2 || report.Calibration.ProbabilityObservations != 4 {
		t.Fatalf("calibration samples = %#v", report.Calibration)
	}
	if math.Abs(report.Calibration.BinaryBrier-0.19) > 1e-12 {
		t.Fatalf("binary Brier = %v, want 0.19", report.Calibration.BinaryBrier)
	}
	if math.Abs(report.Calibration.ExpectedCalibrationError-0.25) > 1e-12 {
		t.Fatalf("ECE = %v, want 0.25", report.Calibration.ExpectedCalibrationError)
	}
}
