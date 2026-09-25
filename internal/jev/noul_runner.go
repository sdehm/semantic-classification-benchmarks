package jev

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/sdehm/semantic-classification-benchmarks/internal/benchmark"
)

const (
	neutralLabel                          = "neutral"
	NoulEmptyPredictionFallbackNeutral    = "fallback-neutral"
	NoulEmptyPredictionHighestProbability = "highest-probability"
	NoulStateTruncationNone               = "none"
	NoulStateTruncationHeadTail           = "head-tail"
)

const omittedStateMarker = "\n\n[... document middle omitted to fit the configured context limit ...]\n\n"

type NoulCriteria struct {
	True  string `json:"true"`
	False string `json:"false"`
}

type NoulRunConfig struct {
	Evaluator                   Evaluator
	Records                     []benchmark.MultiLabelDatasetRecord
	Criteria                    map[string]NoulCriteria
	Model                       string
	Split                       string
	Limit                       int
	SampleSeed                  string
	CriteriaHash                string
	StateField                  string
	QuestionTemplate            string
	EmptyPredictionPolicy       string
	MaxStateCharacters          int
	StateTruncation             string
	Threshold                   float64
	MaxCostUSD                  float64
	InputPricePerMillion        float64
	MaxInputTokensPerRequest    int
	CheckpointPath              string
	OutputPath                  string
	DryRun                      bool
	AllowLegacyCheckpointReplay bool
}

type NoulRunReport struct {
	Model                 string  `json:"model"`
	Split                 string  `json:"split"`
	RequestedLimit        int     `json:"requested_limit"`
	SampleSeed            string  `json:"sample_seed"`
	CriteriaHash          string  `json:"criteria_hash"`
	StateField            string  `json:"state_field"`
	QuestionTemplate      string  `json:"question_template"`
	EmptyPredictionPolicy string  `json:"empty_prediction_policy"`
	MaxStateCharacters    int     `json:"max_state_characters"`
	StateTruncation       string  `json:"state_truncation"`
	Threshold             float64 `json:"threshold"`
	QuestionsPerRequest   int     `json:"questions_per_request"`
	EligibleRecords       int     `json:"eligible_records"`
	SelectedRecords       int     `json:"selected_records"`
	CompletedPredictions  int     `json:"completed_predictions"`
	PlannedRequests       int     `json:"planned_requests"`
	ActualInputTokens     int     `json:"actual_input_tokens"`
	ActualOutputTokens    int     `json:"actual_output_tokens"`
	ActualCostUSD         float64 `json:"actual_cost_usd"`
	MaxCostUSD            float64 `json:"max_cost_usd"`
	ReservedCostPerCall   float64 `json:"reserved_cost_per_call_usd"`
	CheckpointPath        string  `json:"checkpoint_path"`
	OutputPath            string  `json:"output_path"`
	DryRun                bool    `json:"dry_run"`
	LegacyCheckpoint      bool    `json:"legacy_checkpoint"`
}

type noulRawPrediction struct {
	ID                  string             `json:"id"`
	LabelProbabilities  map[string]float64 `json:"label_probabilities"`
	LatencyMilliseconds int64              `json:"latency_ms"`
	Model               string             `json:"model"`
}

type noulCheckpointEntry struct {
	Prediction     noulRawPrediction `json:"prediction"`
	Usage          Usage             `json:"usage"`
	ProvenanceHash string            `json:"provenance_hash"`
}

// RunNoul evaluates all configured label questions in one request per
// record. It checkpoints raw probabilities so threshold changes can replay
// predictions without spending another API call.
func RunNoul(ctx context.Context, config NoulRunConfig) (NoulRunReport, error) {
	config = withNoulRunDefaults(config)
	hash, err := NoulCriteriaHash(config.Criteria)
	if err != nil {
		return NoulRunReport{}, err
	}
	if config.CriteriaHash != "" && config.CriteriaHash != hash {
		return NoulRunReport{}, errors.New("criteria hash does not match the configured Noul criteria")
	}
	config.CriteriaHash = hash
	if err := validateNoulRunConfig(config); err != nil {
		return NoulRunReport{}, err
	}
	eligibleRecords := multiLabelRecordsForSplit(config.Records, config.Split)
	if len(eligibleRecords) == 0 {
		return NoulRunReport{}, fmt.Errorf("no records found for split %q", config.Split)
	}
	if err := validateNoulCriteriaCoverage(eligibleRecords, config.Criteria); err != nil {
		return NoulRunReport{}, err
	}
	records := selectMultiLabelRecords(eligibleRecords, config.Limit, config.SampleSeed)
	fingerprints, err := noulCheckpointFingerprints(config, records)
	if err != nil {
		return NoulRunReport{}, err
	}
	completed, usage, legacy, err := readNoulCheckpoint(config.CheckpointPath, config.Criteria, fingerprints, config.AllowLegacyCheckpointReplay)
	if err != nil {
		return NoulRunReport{}, err
	}
	if config.AllowLegacyCheckpointReplay && !legacy {
		return NoulRunReport{}, errors.New("--allow-legacy-checkpoint-replay requires a complete legacy checkpoint")
	}

	report := noulReportFromUsage(config, len(eligibleRecords), len(records), completed, usage)
	report.LegacyCheckpoint = legacy
	if config.DryRun {
		for _, record := range records {
			if _, exists := completed[record.ID]; exists {
				continue
			}
			if noulExceedsBudget(report.ActualCostUSD, report.PlannedRequests+1, config) {
				break
			}
			report.PlannedRequests++
		}
		return report, nil
	}

	questions := noulQuestions(config.Criteria, config.QuestionTemplate)
	for _, record := range records {
		if _, exists := completed[record.ID]; exists {
			continue
		}
		if config.Evaluator == nil {
			return report, errors.New("a Jev evaluator is required for incomplete Noul runs")
		}
		if noulExceedsBudget(report.ActualCostUSD, 1, config) {
			return report, fmt.Errorf(
				"local budget exhausted before record %q: actual cost $%.6f plus a $%.6f request reservation exceeds $%.6f",
				record.ID,
				report.ActualCostUSD,
				report.ReservedCostPerCall,
				config.MaxCostUSD,
			)
		}

		stateText, err := truncateNoulState(record.Text, config)
		if err != nil {
			return report, fmt.Errorf("prepare state for record %q: %w", record.ID, err)
		}
		started := time.Now()
		response, err := config.Evaluator.Evaluate(ctx, SystemOneRequest{
			State:     map[string]string{config.StateField: stateText},
			Model:     config.Model,
			Questions: questions,
		})
		if err != nil {
			return report, fmt.Errorf("evaluate record %q: %w", record.ID, err)
		}
		rawPrediction, err := rawNoulPrediction(record.ID, response, config.Criteria, time.Since(started))
		if err != nil {
			return report, err
		}
		if response.Usage.InputTokens < 0 || response.Usage.OutputTokens < 0 {
			return report, fmt.Errorf("record %q returned negative token usage", record.ID)
		}
		if response.Usage.InputTokens > config.MaxInputTokensPerRequest {
			return report, fmt.Errorf(
				"record %q used %d input tokens, exceeding the configured per-request reservation of %d",
				record.ID,
				response.Usage.InputTokens,
				config.MaxInputTokensPerRequest,
			)
		}
		if err := appendNoulCheckpoint(
			config.CheckpointPath,
			noulCheckpointEntry{Prediction: rawPrediction, Usage: response.Usage, ProvenanceHash: fingerprints[record.ID]},
		); err != nil {
			return report, err
		}
		completed[rawPrediction.ID] = rawPrediction
		usage.InputTokens += response.Usage.InputTokens
		usage.OutputTokens += response.Usage.OutputTokens
		report = noulReportFromUsage(config, len(eligibleRecords), len(records), completed, usage)
		report.LegacyCheckpoint = legacy
		if err := writeNoulSnapshot(
			config.OutputPath, records, completed, config.Threshold, config.EmptyPredictionPolicy,
		); err != nil {
			return report, err
		}
	}
	if len(completed) > 0 {
		if err := writeNoulSnapshot(
			config.OutputPath, records, completed, config.Threshold, config.EmptyPredictionPolicy,
		); err != nil {
			return report, err
		}
	}
	return report, nil
}

func withNoulRunDefaults(config NoulRunConfig) NoulRunConfig {
	if config.StateField == "" {
		config.StateField = "comment"
	}
	if config.QuestionTemplate == "" {
		config.QuestionTemplate = fmt.Sprintf("Does `%s` express %%s?", config.StateField)
	}
	if config.EmptyPredictionPolicy == "" {
		config.EmptyPredictionPolicy = NoulEmptyPredictionFallbackNeutral
	}
	if config.StateTruncation == "" {
		config.StateTruncation = NoulStateTruncationNone
	}
	return config
}

func truncateNoulState(text string, config NoulRunConfig) (string, error) {
	if config.StateTruncation == NoulStateTruncationNone {
		return text, nil
	}
	runes := []rune(text)
	if len(runes) <= config.MaxStateCharacters {
		return text, nil
	}
	if config.StateTruncation != NoulStateTruncationHeadTail {
		return "", fmt.Errorf("unsupported state truncation policy %q", config.StateTruncation)
	}
	remaining := config.MaxStateCharacters - len([]rune(omittedStateMarker))
	headLength := (remaining + 1) / 2
	tailLength := remaining / 2
	return string(runes[:headLength]) + omittedStateMarker + string(runes[len(runes)-tailLength:]), nil
}

func validateNoulRunConfig(config NoulRunConfig) error {
	if err := benchmark.ValidateMultiLabelDatasetRecords(config.Records); err != nil {
		return err
	}
	if strings.TrimSpace(config.Model) == "" {
		return errors.New("Jev model must not be empty")
	}
	if config.Split != benchmark.SplitTrain && config.Split != benchmark.SplitValidation && config.Split != benchmark.SplitTest {
		return fmt.Errorf("unsupported split %q", config.Split)
	}
	if config.Limit < 1 {
		return fmt.Errorf("record limit must be at least 1, got %d", config.Limit)
	}
	if strings.TrimSpace(config.SampleSeed) == "" {
		return errors.New("sample seed must not be empty")
	}
	if strings.TrimSpace(config.StateField) == "" {
		return errors.New("state field must not be empty")
	}
	if strings.Count(config.QuestionTemplate, "%s") != 1 {
		return errors.New("Noul question template must contain exactly one %s label placeholder")
	}
	switch config.EmptyPredictionPolicy {
	case NoulEmptyPredictionFallbackNeutral, NoulEmptyPredictionHighestProbability:
	default:
		return fmt.Errorf("unsupported empty prediction policy %q", config.EmptyPredictionPolicy)
	}
	if config.MaxStateCharacters < 0 {
		return fmt.Errorf("maximum state characters must not be negative, got %d", config.MaxStateCharacters)
	}
	switch config.StateTruncation {
	case NoulStateTruncationNone:
		if config.MaxStateCharacters != 0 {
			return errors.New("a state character limit requires a truncation policy")
		}
	case NoulStateTruncationHeadTail:
		if config.MaxStateCharacters <= len([]rune(omittedStateMarker)) {
			return fmt.Errorf(
				"maximum state characters must exceed %d for head-tail truncation",
				len([]rune(omittedStateMarker)),
			)
		}
	default:
		return fmt.Errorf("unsupported state truncation policy %q", config.StateTruncation)
	}
	if config.Threshold <= 0 || config.Threshold >= 1 {
		return fmt.Errorf("threshold must be between zero and one, got %f", config.Threshold)
	}
	if config.MaxCostUSD <= 0 {
		return fmt.Errorf("maximum cost must be positive, got %f", config.MaxCostUSD)
	}
	if config.InputPricePerMillion <= 0 {
		return fmt.Errorf("input price must be positive, got %f", config.InputPricePerMillion)
	}
	if config.MaxInputTokensPerRequest < 1 || config.MaxInputTokensPerRequest > MaxInputTokensPerRequest {
		return fmt.Errorf(
			"maximum input tokens per request must be between 1 and %d, got %d",
			MaxInputTokensPerRequest,
			config.MaxInputTokensPerRequest,
		)
	}
	if strings.TrimSpace(config.CheckpointPath) == "" || strings.TrimSpace(config.OutputPath) == "" {
		return errors.New("checkpoint and output paths must not be empty")
	}
	if len(config.Criteria) == 0 {
		return errors.New("at least one Noul label criterion is required")
	}
	for label, criterion := range config.Criteria {
		if strings.TrimSpace(label) == "" || label == neutralLabel {
			return fmt.Errorf("invalid Noul label %q", label)
		}
		if strings.TrimSpace(criterion.True) == "" || strings.TrimSpace(criterion.False) == "" {
			return fmt.Errorf("Noul criterion %q must define true and false meanings", label)
		}
	}
	return nil
}

func multiLabelRecordsForSplit(records []benchmark.MultiLabelDatasetRecord, split string) []benchmark.MultiLabelDatasetRecord {
	selected := make([]benchmark.MultiLabelDatasetRecord, 0)
	for _, record := range records {
		if record.Split == split {
			selected = append(selected, record)
		}
	}
	return selected
}

// selectMultiLabelRecords first covers every available gold label, prioritizing
// rarer labels, then fills the remaining budget by a seed-derived rank.
func selectMultiLabelRecords(records []benchmark.MultiLabelDatasetRecord, limit int, seed string) []benchmark.MultiLabelDatasetRecord {
	if limit >= len(records) {
		return append([]benchmark.MultiLabelDatasetRecord(nil), records...)
	}
	type rankedRecord struct {
		record benchmark.MultiLabelDatasetRecord
		labels []string
		rank   [sha256.Size]byte
	}
	ranked := make([]rankedRecord, 0, len(records))
	labelCounts := make(map[string]int)
	for _, record := range records {
		labels, err := benchmark.DecodeGoldLabels(record.GoldLabelsJSON)
		if err != nil {
			panic(fmt.Sprintf("validated multi-label record %q could not be decoded: %v", record.ID, err))
		}
		for _, label := range labels {
			labelCounts[label]++
		}
		ranked = append(ranked, rankedRecord{
			record: record,
			labels: labels,
			rank:   sha256.Sum256([]byte(seed + "\x00" + record.ID)),
		})
	}
	sort.Slice(ranked, func(left, right int) bool {
		return string(ranked[left].rank[:]) < string(ranked[right].rank[:])
	})
	labels := make([]string, 0, len(labelCounts))
	for label := range labelCounts {
		labels = append(labels, label)
	}
	sort.Slice(labels, func(left, right int) bool {
		if labelCounts[labels[left]] == labelCounts[labels[right]] {
			return labels[left] < labels[right]
		}
		return labelCounts[labels[left]] < labelCounts[labels[right]]
	})

	selected := make(map[string]struct{}, limit)
	for _, label := range labels {
		if len(selected) == limit {
			break
		}
		for _, candidate := range ranked {
			if _, exists := selected[candidate.record.ID]; exists || !containsLabel(candidate.labels, label) {
				continue
			}
			selected[candidate.record.ID] = struct{}{}
			break
		}
	}
	for _, candidate := range ranked {
		if len(selected) == limit {
			break
		}
		selected[candidate.record.ID] = struct{}{}
	}
	result := make([]benchmark.MultiLabelDatasetRecord, 0, len(selected))
	for _, candidate := range ranked {
		if _, exists := selected[candidate.record.ID]; exists {
			result = append(result, candidate.record)
		}
	}
	return result
}

func containsLabel(labels []string, target string) bool {
	for _, label := range labels {
		if label == target {
			return true
		}
	}
	return false
}

func validateNoulCriteriaCoverage(records []benchmark.MultiLabelDatasetRecord, criteria map[string]NoulCriteria) error {
	for _, record := range records {
		labels, err := benchmark.DecodeGoldLabels(record.GoldLabelsJSON)
		if err != nil {
			return fmt.Errorf("decode labels for record %q: %w", record.ID, err)
		}
		for _, label := range labels {
			if label != neutralLabel && strings.TrimSpace(criteria[label].True) == "" {
				return fmt.Errorf("criteria does not define dataset label %q", label)
			}
		}
	}
	return nil
}

func noulQuestions(criteria map[string]NoulCriteria, questionTemplate string) map[string]Question {
	questions := make(map[string]Question, len(criteria))
	for label, criterion := range criteria {
		questions[label] = Question{
			Type:         "noul",
			Instructions: fmt.Sprintf(questionTemplate, label),
			Criteria:     criterion,
		}
	}
	return questions
}

func rawNoulPrediction(id string, response SystemOneResponse, criteria map[string]NoulCriteria, elapsed time.Duration) (noulRawPrediction, error) {
	if strings.TrimSpace(response.Model) == "" {
		return noulRawPrediction{}, fmt.Errorf("record %q returned no model identifier", id)
	}
	if response.Usage.InputTokens < 0 || response.Usage.OutputTokens < 0 {
		return noulRawPrediction{}, fmt.Errorf("record %q returned negative token usage", id)
	}
	probabilities := make(map[string]float64, len(criteria))
	for label := range criteria {
		answer, exists := response.Answers[label]
		if !exists || answer.Type != "noul" || answer.Noul == nil {
			return noulRawPrediction{}, fmt.Errorf("record %q returned no valid Noul answer for %q", id, label)
		}
		if math.IsNaN(*answer.Noul) || math.IsInf(*answer.Noul, 0) || *answer.Noul < 0 || *answer.Noul > 1 {
			return noulRawPrediction{}, fmt.Errorf("record %q returned invalid Noul probability for %q", id, label)
		}
		probabilities[label] = *answer.Noul
	}
	if len(response.Answers) != len(criteria) {
		return noulRawPrediction{}, fmt.Errorf("record %q returned unexpected Noul answers", id)
	}
	return noulRawPrediction{
		ID:                  id,
		LabelProbabilities:  probabilities,
		LatencyMilliseconds: elapsed.Milliseconds(),
		Model:               response.Model,
	}, nil
}

func noulCheckpointFingerprints(config NoulRunConfig, records []benchmark.MultiLabelDatasetRecord) (map[string]string, error) {
	runHash, err := checkpointFingerprint(struct {
		Version            int
		Model              string
		Split              string
		SampleSeed         string
		Criteria           map[string]NoulCriteria
		StateField         string
		QuestionTemplate   string
		MaxStateCharacters int
		StateTruncation    string
		OmissionMarker     string
	}{
		Version:            1,
		Model:              config.Model,
		Split:              config.Split,
		SampleSeed:         config.SampleSeed,
		Criteria:           config.Criteria,
		StateField:         config.StateField,
		QuestionTemplate:   config.QuestionTemplate,
		MaxStateCharacters: config.MaxStateCharacters,
		StateTruncation:    config.StateTruncation,
		OmissionMarker:     omittedStateMarker,
	})
	if err != nil {
		return nil, err
	}
	fingerprints := make(map[string]string, len(records))
	for _, record := range records {
		fingerprints[record.ID], err = checkpointFingerprint(runHash, record)
		if err != nil {
			return nil, err
		}
	}
	return fingerprints, nil
}

func readNoulCheckpoint(
	path string, criteria map[string]NoulCriteria, fingerprints map[string]string, allowLegacy bool,
) (map[string]noulRawPrediction, Usage, bool, error) {
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return make(map[string]noulRawPrediction), Usage{}, false, nil
	}
	if err != nil {
		return nil, Usage{}, false, fmt.Errorf("open Noul checkpoint: %w", err)
	}
	defer file.Close()

	completed := make(map[string]noulRawPrediction)
	var usage Usage
	var legacy, fingerprinted bool
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for line := 1; scanner.Scan(); line++ {
		var entry noulCheckpointEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			return nil, Usage{}, false, fmt.Errorf("decode Noul checkpoint line %d: %w", line, err)
		}
		if err := validateRawNoulPrediction(entry.Prediction, criteria); err != nil {
			return nil, Usage{}, false, fmt.Errorf("validate Noul checkpoint line %d: %w", line, err)
		}
		if _, exists := completed[entry.Prediction.ID]; exists {
			return nil, Usage{}, false, fmt.Errorf("Noul checkpoint has duplicate prediction id %q", entry.Prediction.ID)
		}
		if entry.Usage.InputTokens < 0 || entry.Usage.OutputTokens < 0 {
			return nil, Usage{}, false, fmt.Errorf("Noul checkpoint line %d has negative token usage", line)
		}
		expected, exists := fingerprints[entry.Prediction.ID]
		if !exists {
			return nil, Usage{}, false, fmt.Errorf("Noul checkpoint prediction %q is not in this run's selected records", entry.Prediction.ID)
		}
		if entry.ProvenanceHash == "" {
			legacy = true
			if !allowLegacy {
				return nil, Usage{}, false, fmt.Errorf("Noul checkpoint line %d has no provenance; use --allow-legacy-checkpoint-replay only for a complete, independently verified checkpoint", line)
			}
		} else {
			fingerprinted = true
			if entry.ProvenanceHash != expected {
				return nil, Usage{}, false, fmt.Errorf("Noul checkpoint line %d provenance mismatch for record %q", line, entry.Prediction.ID)
			}
		}
		completed[entry.Prediction.ID] = entry.Prediction
		usage.InputTokens += entry.Usage.InputTokens
		usage.OutputTokens += entry.Usage.OutputTokens
	}
	if err := scanner.Err(); err != nil {
		return nil, Usage{}, false, fmt.Errorf("read Noul checkpoint: %w", err)
	}
	if legacy && fingerprinted {
		return nil, Usage{}, false, errors.New("Noul checkpoint mixes legacy and fingerprinted predictions")
	}
	if legacy && len(completed) != len(fingerprints) {
		return nil, Usage{}, false, errors.New("legacy Noul checkpoint is incomplete; replay is read-only and cannot resume inference")
	}
	return completed, usage, legacy, nil
}

func validateRawNoulPrediction(prediction noulRawPrediction, criteria map[string]NoulCriteria) error {
	if strings.TrimSpace(prediction.ID) == "" {
		return errors.New("prediction has an empty id")
	}
	if strings.TrimSpace(prediction.Model) == "" {
		return fmt.Errorf("prediction %q has an empty model", prediction.ID)
	}
	if prediction.LatencyMilliseconds < 0 {
		return fmt.Errorf("prediction %q has a negative latency", prediction.ID)
	}
	if len(prediction.LabelProbabilities) != len(criteria) {
		return fmt.Errorf("prediction %q has %d probabilities, want %d", prediction.ID, len(prediction.LabelProbabilities), len(criteria))
	}
	for label := range criteria {
		probability, exists := prediction.LabelProbabilities[label]
		if !exists || math.IsNaN(probability) || math.IsInf(probability, 0) || probability < 0 || probability > 1 {
			return fmt.Errorf("prediction %q has invalid probability for %q", prediction.ID, label)
		}
	}
	return nil
}

func appendNoulCheckpoint(path string, entry noulCheckpointEntry) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create Noul checkpoint directory: %w", err)
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open Noul checkpoint: %w", err)
	}
	encoder := json.NewEncoder(file)
	if err := encoder.Encode(entry); err != nil {
		closeErr := file.Close()
		return fmt.Errorf("write Noul checkpoint: %w", errors.Join(err, closeErr))
	}
	if err := file.Sync(); err != nil {
		closeErr := file.Close()
		return fmt.Errorf("sync Noul checkpoint: %w", errors.Join(err, closeErr))
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close Noul checkpoint: %w", err)
	}
	return nil
}

func writeNoulSnapshot(
	path string,
	records []benchmark.MultiLabelDatasetRecord,
	completed map[string]noulRawPrediction,
	threshold float64,
	emptyPredictionPolicy string,
) error {
	predictions := make([]benchmark.MultiLabelPredictionRecord, 0, len(completed))
	for _, record := range records {
		rawPrediction, exists := completed[record.ID]
		if !exists {
			continue
		}
		predictedLabels := make([]string, 0)
		for label, probability := range rawPrediction.LabelProbabilities {
			if probability >= threshold {
				predictedLabels = append(predictedLabels, label)
			}
		}
		if len(predictedLabels) == 0 {
			switch emptyPredictionPolicy {
			case NoulEmptyPredictionFallbackNeutral:
				predictedLabels = []string{neutralLabel}
			case NoulEmptyPredictionHighestProbability:
				predictedLabels = []string{highestProbabilityLabel(rawPrediction.LabelProbabilities)}
			default:
				return fmt.Errorf("unsupported empty prediction policy %q", emptyPredictionPolicy)
			}
		}

		encodedLabels, err := benchmark.EncodeGoldLabels(predictedLabels)
		if err != nil {
			return fmt.Errorf("encode Noul labels for record %q: %w", record.ID, err)
		}
		probabilities, err := json.Marshal(rawPrediction.LabelProbabilities)
		if err != nil {
			return fmt.Errorf("encode Noul probabilities for record %q: %w", record.ID, err)
		}
		predictions = append(predictions, benchmark.MultiLabelPredictionRecord{
			ID:                     record.ID,
			PredictedLabelsJSON:    encodedLabels,
			LabelProbabilitiesJSON: string(probabilities),
			LatencyMilliseconds:    rawPrediction.LatencyMilliseconds,
			Model:                  rawPrediction.Model,
		})
	}
	if err := benchmark.WriteMultiLabelPredictionRecords(path, predictions); err != nil {
		return fmt.Errorf("write Noul prediction snapshot: %w", err)
	}
	return nil
}

func highestProbabilityLabel(probabilities map[string]float64) string {
	label := ""
	probability := -1.0
	for candidate, score := range probabilities {
		if score > probability || (score == probability && (label == "" || candidate < label)) {
			label = candidate
			probability = score
		}
	}
	return label
}

func noulReportFromUsage(config NoulRunConfig, eligibleRecords, selectedRecords int, completed map[string]noulRawPrediction, usage Usage) NoulRunReport {
	return NoulRunReport{
		Model:                 config.Model,
		Split:                 config.Split,
		RequestedLimit:        config.Limit,
		SampleSeed:            config.SampleSeed,
		CriteriaHash:          config.CriteriaHash,
		StateField:            config.StateField,
		QuestionTemplate:      config.QuestionTemplate,
		EmptyPredictionPolicy: config.EmptyPredictionPolicy,
		MaxStateCharacters:    config.MaxStateCharacters,
		StateTruncation:       config.StateTruncation,
		Threshold:             config.Threshold,
		QuestionsPerRequest:   len(config.Criteria),
		EligibleRecords:       eligibleRecords,
		SelectedRecords:       selectedRecords,
		CompletedPredictions:  len(completed),
		ActualInputTokens:     usage.InputTokens,
		ActualOutputTokens:    usage.OutputTokens,
		ActualCostUSD:         float64(usage.InputTokens) / 1_000_000 * config.InputPricePerMillion,
		MaxCostUSD:            config.MaxCostUSD,
		ReservedCostPerCall:   float64(config.MaxInputTokensPerRequest) / 1_000_000 * config.InputPricePerMillion,
		CheckpointPath:        config.CheckpointPath,
		OutputPath:            config.OutputPath,
		DryRun:                config.DryRun,
	}
}

func noulExceedsBudget(actualCost float64, additionalRequests int, config NoulRunConfig) bool {
	reservedCost := float64(config.MaxInputTokensPerRequest) / 1_000_000 * config.InputPricePerMillion
	return actualCost+float64(additionalRequests)*reservedCost > config.MaxCostUSD+math.SmallestNonzeroFloat64
}

func NoulCriteriaHash(criteria map[string]NoulCriteria) (string, error) {
	if len(criteria) == 0 {
		return "", errors.New("at least one Noul label criterion is required")
	}
	labels := make([]string, 0, len(criteria))
	for label := range criteria {
		labels = append(labels, label)
	}
	sort.Strings(labels)
	canonical := make(map[string]NoulCriteria, len(criteria))
	for _, label := range labels {
		canonical[label] = criteria[label]
	}
	encoded, err := json.Marshal(canonical)
	if err != nil {
		return "", fmt.Errorf("encode Noul criteria: %w", err)
	}
	hash := sha256.Sum256(encoded)
	return fmt.Sprintf("%x", hash), nil
}
