package benchmark

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/parquet-go/parquet-go"
)

const (
	SplitTrain      = "train"
	SplitValidation = "validation"
	SplitTest       = "test"
)

// DatasetRecord is the single-label dataset contract used by the initial benchmark.
type DatasetRecord struct {
	ID        string `json:"id" parquet:"id"`
	Text      string `json:"text" parquet:"text"`
	Split     string `json:"split" parquet:"split"`
	GoldLabel string `json:"gold_label" parquet:"gold_label"`
}

// PredictionRecord is emitted by every classifier implementation.
//
// LabelProbabilitiesJSON stores a canonical JSON object so implementations with
// different label counts retain their full probability distribution in Parquet.
type PredictionRecord struct {
	ID                     string  `json:"id" parquet:"id"`
	PredictedLabel         string  `json:"predicted_label" parquet:"predicted_label"`
	LabelProbabilitiesJSON string  `json:"label_probabilities_json" parquet:"label_probabilities_json"`
	Confidence             float64 `json:"confidence" parquet:"confidence"`
	HasConfidence          bool    `json:"has_confidence" parquet:"has_confidence"`
	LatencyMilliseconds    int64   `json:"latency_ms" parquet:"latency_ms"`
	Model                  string  `json:"model" parquet:"model"`
}

func ValidateDatasetRecords(records []DatasetRecord) error {
	seenIDs := make(map[string]struct{}, len(records))

	for index, record := range records {
		if strings.TrimSpace(record.ID) == "" {
			return fmt.Errorf("record %d has an empty id", index)
		}
		if _, exists := seenIDs[record.ID]; exists {
			return fmt.Errorf("duplicate record id %q", record.ID)
		}
		seenIDs[record.ID] = struct{}{}

		if strings.TrimSpace(record.Text) == "" {
			return fmt.Errorf("record %q has empty text", record.ID)
		}
		if !validSplit(record.Split) {
			return fmt.Errorf("record %q has unsupported split %q", record.ID, record.Split)
		}
		if strings.TrimSpace(record.GoldLabel) == "" {
			return fmt.Errorf("record %q has an empty gold label", record.ID)
		}
	}

	return nil
}

func ReadDatasetRecords(path string) ([]DatasetRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open dataset records: %w", err)
	}
	defer file.Close()

	reader := parquet.NewGenericReader[DatasetRecord](file)
	defer reader.Close()

	records := make([]DatasetRecord, 0)
	buffer := make([]DatasetRecord, 256)
	for {
		count, readErr := reader.Read(buffer)
		records = append(records, buffer[:count]...)
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return nil, fmt.Errorf("read dataset records: %w", readErr)
		}
	}

	if err := ValidateDatasetRecords(records); err != nil {
		return nil, err
	}
	return records, nil
}

func WriteDatasetRecords(path string, records []DatasetRecord) error {
	if err := ValidateDatasetRecords(records); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create dataset output directory: %w", err)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create dataset records: %w", err)
	}

	writer := parquet.NewGenericWriter[DatasetRecord](file)
	_, writeErr := writer.Write(records)
	closeWriterErr := writer.Close()
	closeFileErr := file.Close()
	if writeErr != nil || closeWriterErr != nil || closeFileErr != nil {
		return fmt.Errorf("write dataset records: %w", errors.Join(writeErr, closeWriterErr, closeFileErr))
	}
	return nil
}

func ReadPredictionRecords(path string) ([]PredictionRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open prediction records: %w", err)
	}
	defer file.Close()

	reader := parquet.NewGenericReader[PredictionRecord](file)
	defer reader.Close()

	records := make([]PredictionRecord, 0)
	buffer := make([]PredictionRecord, 256)
	for {
		count, readErr := reader.Read(buffer)
		records = append(records, buffer[:count]...)
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return nil, fmt.Errorf("read prediction records: %w", readErr)
		}
	}
	return records, nil
}

func WritePredictionRecords(path string, records []PredictionRecord) error {
	if err := ValidatePredictionRecords(records); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create prediction output directory: %w", err)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create prediction records: %w", err)
	}

	writer := parquet.NewGenericWriter[PredictionRecord](file)
	_, writeErr := writer.Write(records)
	closeWriterErr := writer.Close()
	closeFileErr := file.Close()
	if writeErr != nil || closeWriterErr != nil || closeFileErr != nil {
		return fmt.Errorf("write prediction records: %w", errors.Join(writeErr, closeWriterErr, closeFileErr))
	}
	return nil
}

func ValidatePredictionRecords(records []PredictionRecord) error {
	seenIDs := make(map[string]struct{}, len(records))
	for index, record := range records {
		if strings.TrimSpace(record.ID) == "" {
			return fmt.Errorf("prediction %d has an empty id", index)
		}
		if _, exists := seenIDs[record.ID]; exists {
			return fmt.Errorf("duplicate prediction id %q", record.ID)
		}
		seenIDs[record.ID] = struct{}{}
		if strings.TrimSpace(record.PredictedLabel) == "" {
			return fmt.Errorf("prediction %q has an empty predicted label", record.ID)
		}
		if strings.TrimSpace(record.Model) == "" {
			return fmt.Errorf("prediction %q has an empty model", record.ID)
		}
		if record.LatencyMilliseconds < 0 {
			return fmt.Errorf("prediction %q has a negative latency", record.ID)
		}
	}
	return nil
}

func validSplit(split string) bool {
	return split == SplitTrain || split == SplitValidation || split == SplitTest
}
