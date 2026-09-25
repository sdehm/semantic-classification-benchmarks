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

	"github.com/sdehm/jev-classify-test/internal/benchmark"
)

const (
	MaxInputTokensPerRequest = 64000
	choiceInstructions       = "Classify this customer request using exactly one label from the criteria."
)

type Evaluator interface {
	Evaluate(context.Context, SystemOneRequest) (SystemOneResponse, error)
}

type ChoiceRunConfig struct {
	Evaluator                   Evaluator
	Records                     []benchmark.DatasetRecord
	Criteria                    map[string]string
	Model                       string
	Split                       string
	Limit                       int
	SampleSeed                  string
	CriteriaVariant             string
	CriteriaHash                string
	MaxCostUSD                  float64
	InputPricePerMillion        float64
	MaxInputTokensPerRequest    int
	CheckpointPath              string
	OutputPath                  string
	DryRun                      bool
	AllowLegacyCheckpointReplay bool
}

type ChoiceRunReport struct {
	Model                string  `json:"model"`
	Split                string  `json:"split"`
	RequestedLimit       int     `json:"requested_limit"`
	SampleSeed           string  `json:"sample_seed"`
	CriteriaVariant      string  `json:"criteria_variant"`
	CriteriaHash         string  `json:"criteria_hash"`
	EligibleRecords      int     `json:"eligible_records"`
	SelectedRecords      int     `json:"selected_records"`
	CompletedPredictions int     `json:"completed_predictions"`
	PlannedRequests      int     `json:"planned_requests"`
	ActualInputTokens    int     `json:"actual_input_tokens"`
	ActualOutputTokens   int     `json:"actual_output_tokens"`
	ActualCostUSD        float64 `json:"actual_cost_usd"`
	MaxCostUSD           float64 `json:"max_cost_usd"`
	ReservedCostPerCall  float64 `json:"reserved_cost_per_call_usd"`
	CheckpointPath       string  `json:"checkpoint_path"`
	OutputPath           string  `json:"output_path"`
	DryRun               bool    `json:"dry_run"`
	LegacyCheckpoint     bool    `json:"legacy_checkpoint"`
}

type checkpointEntry struct {
	Prediction     benchmark.PredictionRecord `json:"prediction"`
	Usage          Usage                      `json:"usage"`
	ProvenanceHash string                     `json:"provenance_hash"`
}

// RunChoice evaluates records sequentially. Before every request it reserves
// the documented maximum input context against the caller's local budget.
func RunChoice(ctx context.Context, config ChoiceRunConfig) (ChoiceRunReport, error) {
	if err := validateChoiceRunConfig(&config); err != nil {
		return ChoiceRunReport{}, err
	}
	eligibleRecords := recordsForSplit(config.Records, config.Split)
	if len(eligibleRecords) == 0 {
		return ChoiceRunReport{}, fmt.Errorf("no records found for split %q", config.Split)
	}
	if err := validateCriteriaCoverage(eligibleRecords, config.Criteria); err != nil {
		return ChoiceRunReport{}, err
	}
	records := selectRecords(eligibleRecords, config.Limit, config.SampleSeed)
	fingerprints, err := choiceCheckpointFingerprints(config, records)
	if err != nil {
		return ChoiceRunReport{}, err
	}
	completed, usage, legacy, err := readCheckpoint(config.CheckpointPath, fingerprints, config.AllowLegacyCheckpointReplay)
	if err != nil {
		return ChoiceRunReport{}, err
	}
	if config.AllowLegacyCheckpointReplay && !legacy {
		return ChoiceRunReport{}, errors.New("--allow-legacy-checkpoint-replay requires a complete legacy checkpoint")
	}

	report := reportFromUsage(config, len(eligibleRecords), len(records), completed, usage)
	report.LegacyCheckpoint = legacy
	if config.DryRun {
		for _, record := range records {
			if _, exists := completed[record.ID]; exists {
				continue
			}
			if exceedsBudget(report.ActualCostUSD, report.PlannedRequests+1, config) {
				break
			}
			report.PlannedRequests++
		}
		return report, nil
	}

	questions := map[string]Question{
		"label": {
			Type:         "choice",
			Instructions: choiceInstructions,
			Criteria:     config.Criteria,
		},
	}
	for _, record := range records {
		if _, exists := completed[record.ID]; exists {
			continue
		}
		if config.Evaluator == nil {
			return report, errors.New("a Jev evaluator is required for incomplete Choice runs")
		}
		if exceedsBudget(report.ActualCostUSD, 1, config) {
			return report, fmt.Errorf(
				"local budget exhausted before record %q: actual cost $%.6f plus a $%.6f request reservation exceeds $%.6f",
				record.ID,
				report.ActualCostUSD,
				report.ReservedCostPerCall,
				config.MaxCostUSD,
			)
		}

		started := time.Now()
		response, err := config.Evaluator.Evaluate(ctx, SystemOneRequest{
			State:     record.Text,
			Model:     config.Model,
			Questions: questions,
		})
		if err != nil {
			return report, fmt.Errorf("evaluate record %q: %w", record.ID, err)
		}
		answer, exists := response.Answers["label"]
		if !exists || answer.Type != "choice" || strings.TrimSpace(answer.Choice) == "" {
			return report, fmt.Errorf("record %q returned no valid choice answer", record.ID)
		}
		if _, exists := config.Criteria[answer.Choice]; !exists {
			return report, fmt.Errorf("record %q returned label %q outside the criteria", record.ID, answer.Choice)
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
		probabilities, err := json.Marshal(answer.Probabilities)
		if err != nil {
			return report, fmt.Errorf("encode probabilities for record %q: %w", record.ID, err)
		}
		prediction := benchmark.PredictionRecord{
			ID:                     record.ID,
			PredictedLabel:         answer.Choice,
			LabelProbabilitiesJSON: string(probabilities),
			LatencyMilliseconds:    time.Since(started).Milliseconds(),
			Model:                  response.Model,
		}
		if answer.Confidence != nil {
			prediction.Confidence = *answer.Confidence
			prediction.HasConfidence = true
		}
		if err := appendCheckpoint(config.CheckpointPath, checkpointEntry{
			Prediction: prediction, Usage: response.Usage, ProvenanceHash: fingerprints[record.ID],
		}); err != nil {
			return report, err
		}
		completed[prediction.ID] = prediction
		usage.InputTokens += response.Usage.InputTokens
		usage.OutputTokens += response.Usage.OutputTokens
		report = reportFromUsage(config, len(eligibleRecords), len(records), completed, usage)
		report.LegacyCheckpoint = legacy
		if err := writeSnapshot(config.OutputPath, records, completed); err != nil {
			return report, err
		}
	}
	if len(completed) > 0 {
		if err := writeSnapshot(config.OutputPath, records, completed); err != nil {
			return report, err
		}
	}
	return report, nil
}

func validateChoiceRunConfig(config *ChoiceRunConfig) error {
	if err := benchmark.ValidateDatasetRecords(config.Records); err != nil {
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
		return errors.New("at least one label criterion is required")
	}
	if config.CriteriaVariant == "" {
		config.CriteriaVariant = "unspecified"
	}
	hash, err := CriteriaHash(config.Criteria)
	if err != nil {
		return err
	}
	if config.CriteriaHash != "" && config.CriteriaHash != hash {
		return errors.New("criteria hash does not match the configured Choice criteria")
	}
	config.CriteriaHash = hash
	return nil
}

func recordsForSplit(records []benchmark.DatasetRecord, split string) []benchmark.DatasetRecord {
	selected := make([]benchmark.DatasetRecord, 0)
	for _, record := range records {
		if record.Split == split {
			selected = append(selected, record)
		}
	}
	return selected
}

// selectRecords creates a deterministic, proportionally stratified subset. It
// selects every record when the requested limit covers the split.
func selectRecords(records []benchmark.DatasetRecord, limit int, seed string) []benchmark.DatasetRecord {
	if limit >= len(records) {
		return records
	}

	type rankedRecord struct {
		record benchmark.DatasetRecord
		hash   [sha256.Size]byte
	}
	type labelGroup struct {
		label      string
		records    []rankedRecord
		allocation int
		remainder  float64
	}

	groupsByLabel := make(map[string]*labelGroup)
	for _, record := range records {
		group := groupsByLabel[record.GoldLabel]
		if group == nil {
			group = &labelGroup{label: record.GoldLabel}
			groupsByLabel[record.GoldLabel] = group
		}
		hash := sha256.Sum256([]byte(seed + "\x00" + record.GoldLabel + "\x00" + record.ID))
		group.records = append(group.records, rankedRecord{record: record, hash: hash})
	}

	groups := make([]*labelGroup, 0, len(groupsByLabel))
	allocated := 0
	for _, group := range groupsByLabel {
		target := float64(limit) * float64(len(group.records)) / float64(len(records))
		group.allocation = int(math.Floor(target))
		group.remainder = target - float64(group.allocation)
		allocated += group.allocation
		groups = append(groups, group)
		sort.Slice(group.records, func(left, right int) bool {
			return string(group.records[left].hash[:]) < string(group.records[right].hash[:])
		})
	}
	sort.Slice(groups, func(left, right int) bool {
		if groups[left].remainder == groups[right].remainder {
			return groups[left].label < groups[right].label
		}
		return groups[left].remainder > groups[right].remainder
	})
	for index := 0; allocated < limit; index++ {
		group := groups[index%len(groups)]
		if group.allocation < len(group.records) {
			group.allocation++
			allocated++
		}
	}

	selected := make([]rankedRecord, 0, limit)
	for _, group := range groups {
		selected = append(selected, group.records[:group.allocation]...)
	}
	sort.Slice(selected, func(left, right int) bool {
		return string(selected[left].hash[:]) < string(selected[right].hash[:])
	})
	result := make([]benchmark.DatasetRecord, 0, len(selected))
	for _, selectedRecord := range selected {
		result = append(result, selectedRecord.record)
	}
	return result
}

func validateCriteriaCoverage(records []benchmark.DatasetRecord, criteria map[string]string) error {
	for _, record := range records {
		if strings.TrimSpace(criteria[record.GoldLabel]) == "" {
			return fmt.Errorf("criteria does not define dataset label %q", record.GoldLabel)
		}
	}
	return nil
}

func choiceCheckpointFingerprints(config ChoiceRunConfig, records []benchmark.DatasetRecord) (map[string]string, error) {
	runHash, err := checkpointFingerprint(struct {
		Version         int
		Model           string
		Split           string
		SampleSeed      string
		CriteriaVariant string
		Criteria        map[string]string
		Instructions    string
	}{
		Version:         1,
		Model:           config.Model,
		Split:           config.Split,
		SampleSeed:      config.SampleSeed,
		CriteriaVariant: config.CriteriaVariant,
		Criteria:        config.Criteria,
		Instructions:    choiceInstructions,
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

func readCheckpoint(path string, fingerprints map[string]string, allowLegacy bool) (map[string]benchmark.PredictionRecord, Usage, bool, error) {
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return make(map[string]benchmark.PredictionRecord), Usage{}, false, nil
	}
	if err != nil {
		return nil, Usage{}, false, fmt.Errorf("open Jev checkpoint: %w", err)
	}
	defer file.Close()

	completed := make(map[string]benchmark.PredictionRecord)
	var usage Usage
	var legacy, fingerprinted bool
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for line := 1; scanner.Scan(); line++ {
		var entry checkpointEntry
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			return nil, Usage{}, false, fmt.Errorf("decode Jev checkpoint line %d: %w", line, err)
		}
		if err := benchmark.ValidatePredictionRecords([]benchmark.PredictionRecord{entry.Prediction}); err != nil {
			return nil, Usage{}, false, fmt.Errorf("validate Jev checkpoint line %d: %w", line, err)
		}
		if _, exists := completed[entry.Prediction.ID]; exists {
			return nil, Usage{}, false, fmt.Errorf("Jev checkpoint has duplicate prediction id %q", entry.Prediction.ID)
		}
		if entry.Usage.InputTokens < 0 || entry.Usage.OutputTokens < 0 {
			return nil, Usage{}, false, fmt.Errorf("Jev checkpoint line %d has negative token usage", line)
		}
		expected, exists := fingerprints[entry.Prediction.ID]
		if !exists {
			return nil, Usage{}, false, fmt.Errorf("Jev checkpoint prediction %q is not in this run's selected records", entry.Prediction.ID)
		}
		if entry.ProvenanceHash == "" {
			legacy = true
			if !allowLegacy {
				return nil, Usage{}, false, fmt.Errorf("Jev checkpoint line %d has no provenance; use --allow-legacy-checkpoint-replay only for a complete, independently verified checkpoint", line)
			}
		} else {
			fingerprinted = true
			if entry.ProvenanceHash != expected {
				return nil, Usage{}, false, fmt.Errorf("Jev checkpoint line %d provenance mismatch for record %q", line, entry.Prediction.ID)
			}
		}
		completed[entry.Prediction.ID] = entry.Prediction
		usage.InputTokens += entry.Usage.InputTokens
		usage.OutputTokens += entry.Usage.OutputTokens
	}
	if err := scanner.Err(); err != nil {
		return nil, Usage{}, false, fmt.Errorf("read Jev checkpoint: %w", err)
	}
	if legacy && fingerprinted {
		return nil, Usage{}, false, errors.New("Jev checkpoint mixes legacy and fingerprinted predictions")
	}
	if legacy && len(completed) != len(fingerprints) {
		return nil, Usage{}, false, errors.New("legacy Jev checkpoint is incomplete; replay is read-only and cannot resume inference")
	}
	return completed, usage, legacy, nil
}

func appendCheckpoint(path string, entry checkpointEntry) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create Jev checkpoint directory: %w", err)
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open Jev checkpoint: %w", err)
	}
	encoder := json.NewEncoder(file)
	if err := encoder.Encode(entry); err != nil {
		closeErr := file.Close()
		return fmt.Errorf("write Jev checkpoint: %w", errors.Join(err, closeErr))
	}
	if err := file.Sync(); err != nil {
		closeErr := file.Close()
		return fmt.Errorf("sync Jev checkpoint: %w", errors.Join(err, closeErr))
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close Jev checkpoint: %w", err)
	}
	return nil
}

func writeSnapshot(path string, records []benchmark.DatasetRecord, completed map[string]benchmark.PredictionRecord) error {
	predictions := make([]benchmark.PredictionRecord, 0, len(completed))
	for _, record := range records {
		if prediction, exists := completed[record.ID]; exists {
			predictions = append(predictions, prediction)
		}
	}
	if err := benchmark.WritePredictionRecords(path, predictions); err != nil {
		return fmt.Errorf("write Jev prediction snapshot: %w", err)
	}
	return nil
}

func reportFromUsage(config ChoiceRunConfig, eligibleRecords, selectedRecords int, completed map[string]benchmark.PredictionRecord, usage Usage) ChoiceRunReport {
	return ChoiceRunReport{
		Model:                config.Model,
		Split:                config.Split,
		RequestedLimit:       config.Limit,
		SampleSeed:           config.SampleSeed,
		CriteriaVariant:      config.CriteriaVariant,
		CriteriaHash:         config.CriteriaHash,
		EligibleRecords:      eligibleRecords,
		SelectedRecords:      selectedRecords,
		CompletedPredictions: len(completed),
		ActualInputTokens:    usage.InputTokens,
		ActualOutputTokens:   usage.OutputTokens,
		ActualCostUSD:        float64(usage.InputTokens) / 1_000_000 * config.InputPricePerMillion,
		MaxCostUSD:           config.MaxCostUSD,
		ReservedCostPerCall:  float64(config.MaxInputTokensPerRequest) / 1_000_000 * config.InputPricePerMillion,
		CheckpointPath:       config.CheckpointPath,
		OutputPath:           config.OutputPath,
		DryRun:               config.DryRun,
	}
}

func exceedsBudget(actualCost float64, additionalRequests int, config ChoiceRunConfig) bool {
	reservedCost := float64(config.MaxInputTokensPerRequest) / 1_000_000 * config.InputPricePerMillion
	return actualCost+float64(additionalRequests)*reservedCost > config.MaxCostUSD+math.SmallestNonzeroFloat64
}
