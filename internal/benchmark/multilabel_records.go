package benchmark

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/parquet-go/parquet-go"
)

// MultiLabelDatasetRecord is the benchmark contract for examples with one or
// more gold labels. GoldLabelsJSON is a canonical, sorted JSON string array.
type MultiLabelDatasetRecord struct {
	ID             string `json:"id" parquet:"id"`
	Text           string `json:"text" parquet:"text"`
	Split          string `json:"split" parquet:"split"`
	GoldLabelsJSON string `json:"gold_labels_json" parquet:"gold_labels_json"`
}

// MultiLabelPredictionRecord is emitted by multi-label classifiers. The label
// arrays use the same canonical representation as MultiLabelDatasetRecord.
type MultiLabelPredictionRecord struct {
	ID                     string `json:"id" parquet:"id"`
	PredictedLabelsJSON    string `json:"predicted_labels_json" parquet:"predicted_labels_json"`
	LabelProbabilitiesJSON string `json:"label_probabilities_json" parquet:"label_probabilities_json"`
	LatencyMilliseconds    int64  `json:"latency_ms" parquet:"latency_ms"`
	Model                  string `json:"model" parquet:"model"`
}

func ValidateMultiLabelDatasetRecords(records []MultiLabelDatasetRecord) error {
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
		if _, err := DecodeGoldLabels(record.GoldLabelsJSON); err != nil {
			return fmt.Errorf("record %q has invalid gold labels: %w", record.ID, err)
		}
	}
	return nil
}

func DecodeGoldLabels(encoded string) ([]string, error) {
	var labels []string
	if err := json.Unmarshal([]byte(encoded), &labels); err != nil {
		return nil, fmt.Errorf("decode JSON: %w", err)
	}
	if len(labels) == 0 {
		return nil, errors.New("must contain at least one label")
	}
	for index, label := range labels {
		if strings.TrimSpace(label) == "" {
			return nil, fmt.Errorf("label %d is empty", index)
		}
		if index > 0 && labels[index-1] >= label {
			return nil, errors.New("labels must be unique and sorted")
		}
	}
	return labels, nil
}

func EncodeGoldLabels(labels []string) (string, error) {
	if len(labels) == 0 {
		return "", errors.New("must contain at least one label")
	}
	canonical := append([]string(nil), labels...)
	sort.Strings(canonical)
	for index, label := range canonical {
		if strings.TrimSpace(label) == "" {
			return "", fmt.Errorf("label %d is empty", index)
		}
		if index > 0 && canonical[index-1] == label {
			return "", fmt.Errorf("duplicate label %q", label)
		}
	}
	encoded, err := json.Marshal(canonical)
	if err != nil {
		return "", fmt.Errorf("encode labels: %w", err)
	}
	return string(encoded), nil
}

func ReadMultiLabelDatasetRecords(path string) ([]MultiLabelDatasetRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open multi-label dataset records: %w", err)
	}
	defer file.Close()

	reader := parquet.NewGenericReader[MultiLabelDatasetRecord](file)
	defer reader.Close()

	records := make([]MultiLabelDatasetRecord, 0)
	buffer := make([]MultiLabelDatasetRecord, 256)
	for {
		count, readErr := reader.Read(buffer)
		records = append(records, buffer[:count]...)
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return nil, fmt.Errorf("read multi-label dataset records: %w", readErr)
		}
	}
	if err := ValidateMultiLabelDatasetRecords(records); err != nil {
		return nil, err
	}
	return records, nil
}

func WriteMultiLabelDatasetRecords(path string, records []MultiLabelDatasetRecord) error {
	if err := ValidateMultiLabelDatasetRecords(records); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create multi-label dataset output directory: %w", err)
	}
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create multi-label dataset records: %w", err)
	}
	writer := parquet.NewGenericWriter[MultiLabelDatasetRecord](file)
	_, writeErr := writer.Write(records)
	closeWriterErr := writer.Close()
	closeFileErr := file.Close()
	if writeErr != nil || closeWriterErr != nil || closeFileErr != nil {
		return fmt.Errorf(
			"write multi-label dataset records: %w",
			errors.Join(writeErr, closeWriterErr, closeFileErr),
		)
	}
	return nil
}

func ReadMultiLabelPredictionRecords(path string) ([]MultiLabelPredictionRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open multi-label prediction records: %w", err)
	}
	defer file.Close()

	reader := parquet.NewGenericReader[MultiLabelPredictionRecord](file)
	defer reader.Close()

	records := make([]MultiLabelPredictionRecord, 0)
	buffer := make([]MultiLabelPredictionRecord, 256)
	for {
		count, readErr := reader.Read(buffer)
		records = append(records, buffer[:count]...)
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			return nil, fmt.Errorf("read multi-label prediction records: %w", readErr)
		}
	}
	if err := ValidateMultiLabelPredictionRecords(records); err != nil {
		return nil, err
	}
	return records, nil
}

func WriteMultiLabelPredictionRecords(path string, records []MultiLabelPredictionRecord) error {
	if err := ValidateMultiLabelPredictionRecords(records); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create multi-label prediction output directory: %w", err)
	}
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create multi-label prediction records: %w", err)
	}
	writer := parquet.NewGenericWriter[MultiLabelPredictionRecord](file)
	_, writeErr := writer.Write(records)
	closeWriterErr := writer.Close()
	closeFileErr := file.Close()
	if writeErr != nil || closeWriterErr != nil || closeFileErr != nil {
		return fmt.Errorf(
			"write multi-label prediction records: %w",
			errors.Join(writeErr, closeWriterErr, closeFileErr),
		)
	}
	return nil
}

func ValidateMultiLabelPredictionRecords(records []MultiLabelPredictionRecord) error {
	seenIDs := make(map[string]struct{}, len(records))
	for index, record := range records {
		if strings.TrimSpace(record.ID) == "" {
			return fmt.Errorf("prediction %d has an empty id", index)
		}
		if _, exists := seenIDs[record.ID]; exists {
			return fmt.Errorf("duplicate prediction id %q", record.ID)
		}
		seenIDs[record.ID] = struct{}{}
		if _, err := DecodeGoldLabels(record.PredictedLabelsJSON); err != nil {
			return fmt.Errorf("prediction %q has invalid predicted labels: %w", record.ID, err)
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
