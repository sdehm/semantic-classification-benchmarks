package benchmark

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

type LabelMetrics struct {
	Label     string  `json:"label"`
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
	Support   int     `json:"support"`
}

type MetricReport struct {
	Split           string             `json:"split"`
	Examples        int                `json:"examples"`
	Scored          int                `json:"scored_examples"`
	Missing         int                `json:"missing_predictions"`
	Coverage        float64            `json:"coverage"`
	Complete        bool               `json:"complete"`
	Accuracy        float64            `json:"accuracy"`
	MacroF1         float64            `json:"macro_f1"`
	Latency         LatencySummary     `json:"latency"`
	Calibration     CalibrationSummary `json:"calibration"`
	PerLabelMetrics []LabelMetrics     `json:"per_label_metrics"`
}

type LatencySummary struct {
	Samples       int   `json:"samples"`
	TotalMillis   int64 `json:"total_ms"`
	MinimumMillis int64 `json:"min_ms"`
	P50Millis     int64 `json:"p50_ms"`
	P95Millis     int64 `json:"p95_ms"`
	MaximumMillis int64 `json:"max_ms"`
}

type CalibrationSummary struct {
	ConfidenceSamples        int              `json:"confidence_samples"`
	ProbabilitySamples       int              `json:"probability_samples"`
	MulticlassBrier          float64          `json:"multiclass_brier"`
	ExpectedCalibrationError float64          `json:"expected_calibration_error"`
	Bins                     []CalibrationBin `json:"bins"`
}

type CalibrationBin struct {
	LowerBound     float64 `json:"lower_bound"`
	UpperBound     float64 `json:"upper_bound"`
	Examples       int     `json:"examples"`
	Accuracy       float64 `json:"accuracy"`
	MeanConfidence float64 `json:"mean_confidence"`
}

// EvaluateSingleLabel calculates metrics only from completed predictions for
// records in split. Coverage and completion identify partial benchmark runs.
func EvaluateSingleLabel(records []DatasetRecord, predictions []PredictionRecord, split string) (MetricReport, error) {
	if err := ValidateDatasetRecords(records); err != nil {
		return MetricReport{}, err
	}
	if err := ValidatePredictionRecords(predictions); err != nil {
		return MetricReport{}, err
	}
	if !validSplit(split) {
		return MetricReport{}, fmt.Errorf("unsupported split %q", split)
	}

	predictionsByID := make(map[string]PredictionRecord, len(predictions))
	for _, prediction := range predictions {
		predictionsByID[prediction.ID] = prediction
	}

	type counts struct{ truePositive, falsePositive, falseNegative, support int }
	countsByLabel := make(map[string]*counts)
	getCounts := func(label string) *counts {
		if countsByLabel[label] == nil {
			countsByLabel[label] = &counts{}
		}
		return countsByLabel[label]
	}

	report := MetricReport{Split: split}
	splitLabels := make(map[string]struct{})
	for _, record := range records {
		if record.Split == split {
			splitLabels[record.GoldLabel] = struct{}{}
		}
	}
	var calibration calibrationAccumulator
	correct := 0
	latencies := make([]int64, 0)
	for _, record := range records {
		if record.Split != split {
			continue
		}
		report.Examples++

		prediction, found := predictionsByID[record.ID]
		if !found {
			report.Missing++
			continue
		}
		report.Scored++
		getCounts(record.GoldLabel).support++
		latencies = append(latencies, prediction.LatencyMilliseconds)

		isCorrect := prediction.PredictedLabel == record.GoldLabel
		if err := calibration.add(prediction, record.GoldLabel, splitLabels, isCorrect); err != nil {
			return MetricReport{}, fmt.Errorf("evaluate calibration for prediction %q: %w", prediction.ID, err)
		}

		if isCorrect {
			correct++
			getCounts(record.GoldLabel).truePositive++
			continue
		}
		getCounts(record.GoldLabel).falseNegative++
		getCounts(prediction.PredictedLabel).falsePositive++
	}
	if report.Examples == 0 {
		return MetricReport{}, fmt.Errorf("no records found for split %q", split)
	}
	report.Coverage = float64(report.Scored) / float64(report.Examples)
	report.Complete = report.Missing == 0
	if report.Scored == 0 {
		return report, nil
	}
	report.Accuracy = float64(correct) / float64(report.Scored)

	labels := make([]string, 0, len(countsByLabel))
	for label := range countsByLabel {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	for _, label := range labels {
		count := countsByLabel[label]
		precision := safeDivide(count.truePositive, count.truePositive+count.falsePositive)
		recall := safeDivide(count.truePositive, count.truePositive+count.falseNegative)
		f1 := safeDivideFloat(2*precision*recall, precision+recall)
		report.PerLabelMetrics = append(report.PerLabelMetrics, LabelMetrics{
			Label: label, Precision: precision, Recall: recall, F1: f1, Support: count.support,
		})
		report.MacroF1 += f1
	}
	if len(report.PerLabelMetrics) > 0 {
		report.MacroF1 /= float64(len(report.PerLabelMetrics))
	}
	report.Latency = summarizeLatency(latencies)
	report.Calibration = calibration.summary(len(splitLabels))
	return report, nil
}

const calibrationBinCount = 10

type calibrationAccumulator struct {
	confidenceSamples  int
	probabilitySamples int
	brierTotal         float64
	binCounts          [calibrationBinCount]int
	binCorrect         [calibrationBinCount]int
	binConfidence      [calibrationBinCount]float64
}

func (accumulator *calibrationAccumulator) add(prediction PredictionRecord, goldLabel string, labels map[string]struct{}, correct bool) error {
	if prediction.HasConfidence {
		if math.IsNaN(prediction.Confidence) || math.IsInf(prediction.Confidence, 0) ||
			prediction.Confidence < 0 || prediction.Confidence > 1 {
			return fmt.Errorf("confidence must be in [0, 1], got %v", prediction.Confidence)
		}
		bin := min(int(prediction.Confidence*calibrationBinCount), calibrationBinCount-1)
		accumulator.confidenceSamples++
		accumulator.binCounts[bin]++
		accumulator.binConfidence[bin] += prediction.Confidence
		if correct {
			accumulator.binCorrect[bin]++
		}
	}

	if strings.TrimSpace(prediction.LabelProbabilitiesJSON) == "" {
		return nil
	}
	var probabilities map[string]float64
	if err := json.Unmarshal([]byte(prediction.LabelProbabilitiesJSON), &probabilities); err != nil {
		return fmt.Errorf("decode label probabilities: %w", err)
	}
	var squaredError float64
	for label := range labels {
		probability := probabilities[label]
		if math.IsNaN(probability) || math.IsInf(probability, 0) ||
			probability < 0 || probability > 1 {
			return fmt.Errorf("probability for label %q must be in [0, 1], got %v", label, probability)
		}
		target := 0.0
		if label == goldLabel {
			target = 1
		}
		squaredError += math.Pow(probability-target, 2)
	}
	accumulator.probabilitySamples++
	accumulator.brierTotal += squaredError
	return nil
}

func (accumulator calibrationAccumulator) summary(labelCount int) CalibrationSummary {
	summary := CalibrationSummary{
		ConfidenceSamples:  accumulator.confidenceSamples,
		ProbabilitySamples: accumulator.probabilitySamples,
		Bins:               make([]CalibrationBin, calibrationBinCount),
	}
	if accumulator.probabilitySamples > 0 && labelCount > 0 {
		summary.MulticlassBrier = accumulator.brierTotal / float64(accumulator.probabilitySamples*labelCount)
	}
	for index := range summary.Bins {
		bin := CalibrationBin{
			LowerBound: float64(index) / calibrationBinCount,
			UpperBound: float64(index+1) / calibrationBinCount,
			Examples:   accumulator.binCounts[index],
		}
		if bin.Examples > 0 {
			bin.Accuracy = float64(accumulator.binCorrect[index]) / float64(bin.Examples)
			bin.MeanConfidence = accumulator.binConfidence[index] / float64(bin.Examples)
			summary.ExpectedCalibrationError += math.Abs(bin.Accuracy-bin.MeanConfidence) *
				float64(bin.Examples) / float64(accumulator.confidenceSamples)
		}
		summary.Bins[index] = bin
	}
	return summary
}

func summarizeLatency(latencies []int64) LatencySummary {
	if len(latencies) == 0 {
		return LatencySummary{}
	}
	sorted := append([]int64(nil), latencies...)
	sort.Slice(sorted, func(left, right int) bool { return sorted[left] < sorted[right] })
	var total int64
	for _, latency := range sorted {
		total += latency
	}
	return LatencySummary{
		Samples:       len(sorted),
		TotalMillis:   total,
		MinimumMillis: sorted[0],
		P50Millis:     nearestRank(sorted, 0.50),
		P95Millis:     nearestRank(sorted, 0.95),
		MaximumMillis: sorted[len(sorted)-1],
	}
}

func nearestRank(sorted []int64, percentile float64) int64 {
	index := int(math.Ceil(percentile*float64(len(sorted)))) - 1
	return sorted[index]
}

func safeDivide(numerator, denominator int) float64 {
	if denominator == 0 {
		return 0
	}
	return float64(numerator) / float64(denominator)
}

func safeDivideFloat(numerator, denominator float64) float64 {
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}
