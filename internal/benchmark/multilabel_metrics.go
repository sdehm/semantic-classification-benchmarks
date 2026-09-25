package benchmark

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

type MultiLabelMetricReport struct {
	Split           string                  `json:"split"`
	Examples        int                     `json:"examples"`
	Scored          int                     `json:"scored_examples"`
	Missing         int                     `json:"missing_predictions"`
	Coverage        float64                 `json:"coverage"`
	Complete        bool                    `json:"complete"`
	ExactMatchRatio float64                 `json:"exact_match_ratio"`
	Micro           PrecisionRecallF1       `json:"micro"`
	Macro           PrecisionRecallF1       `json:"macro"`
	PerLabelMetrics []MultiLabelLabelMetric `json:"per_label_metrics"`
	Latency         LatencySummary          `json:"latency"`
	Calibration     MultiLabelCalibration   `json:"calibration"`
}

type PrecisionRecallF1 struct {
	Precision float64 `json:"precision"`
	Recall    float64 `json:"recall"`
	F1        float64 `json:"f1"`
}

type MultiLabelLabelMetric struct {
	Label string `json:"label"`
	PrecisionRecallF1
	Support int `json:"support"`
}

type MultiLabelCalibration struct {
	ProbabilityLabels        []string               `json:"probability_labels"`
	ProbabilitySamples       int                    `json:"probability_samples"`
	ProbabilityObservations  int                    `json:"probability_observations"`
	BinaryBrier              float64                `json:"binary_brier"`
	ExpectedCalibrationError float64                `json:"expected_calibration_error"`
	Bins                     []BinaryCalibrationBin `json:"bins"`
}

type BinaryCalibrationBin struct {
	LowerBound      float64 `json:"lower_bound"`
	UpperBound      float64 `json:"upper_bound"`
	Observations    int     `json:"observations"`
	PositiveRate    float64 `json:"positive_rate"`
	MeanProbability float64 `json:"mean_probability"`
}

// EvaluateMultiLabel calculates exact-match, micro, and macro metrics from
// completed predictions. It uses the complete dataset label universe so a label
// absent from one split still contributes a zero-score macro term if predicted.
func EvaluateMultiLabel(records []MultiLabelDatasetRecord, predictions []MultiLabelPredictionRecord, split string) (MultiLabelMetricReport, error) {
	if err := ValidateMultiLabelDatasetRecords(records); err != nil {
		return MultiLabelMetricReport{}, err
	}
	if err := ValidateMultiLabelPredictionRecords(predictions); err != nil {
		return MultiLabelMetricReport{}, err
	}
	if !validSplit(split) {
		return MultiLabelMetricReport{}, fmt.Errorf("unsupported split %q", split)
	}

	predictionsByID := make(map[string]MultiLabelPredictionRecord, len(predictions))
	for _, prediction := range predictions {
		predictionsByID[prediction.ID] = prediction
	}
	labelUniverse := make(map[string]struct{})
	for _, record := range records {
		labels, err := DecodeGoldLabels(record.GoldLabelsJSON)
		if err != nil {
			return MultiLabelMetricReport{}, fmt.Errorf("decode labels for record %q: %w", record.ID, err)
		}
		for _, label := range labels {
			labelUniverse[label] = struct{}{}
		}
	}

	type counts struct{ truePositive, falsePositive, falseNegative, support int }
	countsByLabel := make(map[string]*counts, len(labelUniverse))
	for label := range labelUniverse {
		countsByLabel[label] = &counts{}
	}

	report := MultiLabelMetricReport{Split: split}
	var exactMatches, truePositive, falsePositive, falseNegative int
	latencies := make([]int64, 0)
	var calibration multiLabelCalibrationAccumulator
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
		goldLabels, err := DecodeGoldLabels(record.GoldLabelsJSON)
		if err != nil {
			return MultiLabelMetricReport{}, fmt.Errorf("decode gold labels for record %q: %w", record.ID, err)
		}
		predictedLabels, err := DecodeGoldLabels(prediction.PredictedLabelsJSON)
		if err != nil {
			return MultiLabelMetricReport{}, fmt.Errorf("decode predicted labels for record %q: %w", record.ID, err)
		}
		goldSet := labelSet(goldLabels)
		predictedSet := labelSet(predictedLabels)
		for label := range predictedSet {
			if _, exists := labelUniverse[label]; !exists {
				return MultiLabelMetricReport{}, fmt.Errorf("prediction %q contains unknown label %q", record.ID, label)
			}
		}
		report.Scored++
		latencies = append(latencies, prediction.LatencyMilliseconds)
		if err := calibration.add(prediction, goldSet, labelUniverse); err != nil {
			return MultiLabelMetricReport{}, fmt.Errorf(
				"evaluate calibration for prediction %q: %w", record.ID, err,
			)
		}
		if equalLabelSets(goldSet, predictedSet) {
			exactMatches++
		}
		for label := range labelUniverse {
			_, gold := goldSet[label]
			_, predicted := predictedSet[label]
			count := countsByLabel[label]
			if gold {
				count.support++
			}
			switch {
			case gold && predicted:
				truePositive++
				count.truePositive++
			case gold:
				falseNegative++
				count.falseNegative++
			case predicted:
				falsePositive++
				count.falsePositive++
			}
		}
	}
	if report.Examples == 0 {
		return MultiLabelMetricReport{}, fmt.Errorf("no records found for split %q", split)
	}
	report.Coverage = float64(report.Scored) / float64(report.Examples)
	report.Complete = report.Missing == 0
	if report.Scored == 0 {
		return report, nil
	}
	report.ExactMatchRatio = float64(exactMatches) / float64(report.Scored)
	report.Micro = precisionRecallF1(truePositive, falsePositive, falseNegative)

	labels := make([]string, 0, len(labelUniverse))
	for label := range labelUniverse {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	for _, label := range labels {
		count := countsByLabel[label]
		metrics := precisionRecallF1(count.truePositive, count.falsePositive, count.falseNegative)
		report.Macro.Precision += metrics.Precision
		report.Macro.Recall += metrics.Recall
		report.Macro.F1 += metrics.F1
		report.PerLabelMetrics = append(report.PerLabelMetrics, MultiLabelLabelMetric{
			Label:             label,
			PrecisionRecallF1: metrics,
			Support:           count.support,
		})
	}
	report.Macro.Precision /= float64(len(labels))
	report.Macro.Recall /= float64(len(labels))
	report.Macro.F1 /= float64(len(labels))
	report.Latency = summarizeLatency(latencies)
	report.Calibration = calibration.summary()
	return report, nil
}

func precisionRecallF1(truePositive, falsePositive, falseNegative int) PrecisionRecallF1 {
	precision := safeDivide(truePositive, truePositive+falsePositive)
	recall := safeDivide(truePositive, truePositive+falseNegative)
	return PrecisionRecallF1{
		Precision: precision,
		Recall:    recall,
		F1:        safeDivideFloat(2*precision*recall, precision+recall),
	}
}

func labelSet(labels []string) map[string]struct{} {
	set := make(map[string]struct{}, len(labels))
	for _, label := range labels {
		set[label] = struct{}{}
	}
	return set
}

func equalLabelSets(left, right map[string]struct{}) bool {
	if len(left) != len(right) {
		return false
	}
	for label := range left {
		if _, exists := right[label]; !exists {
			return false
		}
	}
	return true
}

type multiLabelCalibrationAccumulator struct {
	labels             map[string]struct{}
	probabilitySamples int
	observations       int
	brierTotal         float64
	binCounts          [calibrationBinCount]int
	binPositive        [calibrationBinCount]int
	binProbability     [calibrationBinCount]float64
}

func (accumulator *multiLabelCalibrationAccumulator) add(
	prediction MultiLabelPredictionRecord,
	goldLabels map[string]struct{},
	labelUniverse map[string]struct{},
) error {
	if strings.TrimSpace(prediction.LabelProbabilitiesJSON) == "" {
		return nil
	}
	var probabilities map[string]float64
	if err := json.Unmarshal([]byte(prediction.LabelProbabilitiesJSON), &probabilities); err != nil {
		return fmt.Errorf("decode label probabilities: %w", err)
	}
	if len(probabilities) == 0 {
		return fmt.Errorf("label probabilities must not be empty")
	}
	if accumulator.labels == nil {
		accumulator.labels = make(map[string]struct{}, len(probabilities))
		for label := range probabilities {
			if _, exists := labelUniverse[label]; !exists {
				return fmt.Errorf("probability label %q is not in the dataset label universe", label)
			}
			accumulator.labels[label] = struct{}{}
		}
	} else {
		if len(probabilities) != len(accumulator.labels) {
			return fmt.Errorf("probability label count %d differs from prior count %d", len(probabilities), len(accumulator.labels))
		}
		for label := range accumulator.labels {
			if _, exists := probabilities[label]; !exists {
				return fmt.Errorf("probabilities do not include prior label %q", label)
			}
		}
	}
	for label, probability := range probabilities {
		if math.IsNaN(probability) || math.IsInf(probability, 0) || probability < 0 || probability > 1 {
			return fmt.Errorf("probability for label %q must be in [0, 1], got %v", label, probability)
		}
		target := 0.0
		if _, exists := goldLabels[label]; exists {
			target = 1
		}
		accumulator.brierTotal += math.Pow(probability-target, 2)
		bin := min(int(probability*calibrationBinCount), calibrationBinCount-1)
		accumulator.binCounts[bin]++
		accumulator.binProbability[bin] += probability
		if target == 1 {
			accumulator.binPositive[bin]++
		}
		accumulator.observations++
	}
	accumulator.probabilitySamples++
	return nil
}

func (accumulator multiLabelCalibrationAccumulator) summary() MultiLabelCalibration {
	summary := MultiLabelCalibration{
		ProbabilitySamples:      accumulator.probabilitySamples,
		ProbabilityObservations: accumulator.observations,
		Bins:                    make([]BinaryCalibrationBin, calibrationBinCount),
	}
	for label := range accumulator.labels {
		summary.ProbabilityLabels = append(summary.ProbabilityLabels, label)
	}
	sort.Strings(summary.ProbabilityLabels)
	if accumulator.observations > 0 {
		summary.BinaryBrier = accumulator.brierTotal / float64(accumulator.observations)
	}
	for index := range summary.Bins {
		bin := BinaryCalibrationBin{
			LowerBound:   float64(index) / calibrationBinCount,
			UpperBound:   float64(index+1) / calibrationBinCount,
			Observations: accumulator.binCounts[index],
		}
		if bin.Observations > 0 {
			bin.PositiveRate = float64(accumulator.binPositive[index]) / float64(bin.Observations)
			bin.MeanProbability = accumulator.binProbability[index] / float64(bin.Observations)
			summary.ExpectedCalibrationError += math.Abs(bin.PositiveRate-bin.MeanProbability) *
				float64(bin.Observations) / float64(accumulator.observations)
		}
		summary.Bins[index] = bin
	}
	return summary
}
