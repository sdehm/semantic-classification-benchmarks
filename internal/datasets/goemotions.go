package datasets

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/sdehm/jev-classify-test/internal/benchmark"
)

type GoEmotionsManifest struct {
	ID                string `json:"id"`
	License           string `json:"license"`
	SourceRepository  string `json:"source_repository"`
	SourceRevision    string `json:"source_revision"`
	TrainURL          string `json:"train_url"`
	TrainBlob         string `json:"train_blob"`
	TrainRecords      int    `json:"train_records"`
	ValidationURL     string `json:"validation_url"`
	ValidationBlob    string `json:"validation_blob"`
	ValidationRecords int    `json:"validation_records"`
	TestURL           string `json:"test_url"`
	TestBlob          string `json:"test_blob"`
	TestRecords       int    `json:"test_records"`
	LabelsURL         string `json:"labels_url"`
	LabelsBlob        string `json:"labels_blob"`
	LabelCount        int    `json:"label_count"`
	Citation          string `json:"citation"`
}

func LoadGoEmotionsManifest(path string) (GoEmotionsManifest, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return GoEmotionsManifest{}, fmt.Errorf("read GoEmotions manifest: %w", err)
	}
	var manifest GoEmotionsManifest
	if err := json.Unmarshal(content, &manifest); err != nil {
		return GoEmotionsManifest{}, fmt.Errorf("decode GoEmotions manifest: %w", err)
	}
	if err := validateGoEmotionsManifest(manifest); err != nil {
		return GoEmotionsManifest{}, err
	}
	return manifest, nil
}

// PrepareGoEmotions downloads and verifies the published simplified splits.
// The source data stays local; only normalized benchmark records are written.
func PrepareGoEmotions(ctx context.Context, client *http.Client, manifest GoEmotionsManifest) ([]benchmark.MultiLabelDatasetRecord, error) {
	if err := validateGoEmotionsManifest(manifest); err != nil {
		return nil, err
	}
	if client == nil {
		client = http.DefaultClient
	}
	labelsContent, err := fetchGitBlobContent(ctx, client, manifest.LabelsURL, manifest.LabelsBlob)
	if err != nil {
		return nil, fmt.Errorf("download labels: %w", err)
	}
	labels, err := parseGoEmotionsLabels(labelsContent)
	if err != nil {
		return nil, err
	}
	if len(labels) != manifest.LabelCount {
		return nil, fmt.Errorf("GoEmotions label count = %d, want %d", len(labels), manifest.LabelCount)
	}

	type sourceSplit struct {
		url      string
		blob     string
		split    string
		expected int
	}
	splits := []sourceSplit{
		{manifest.TrainURL, manifest.TrainBlob, benchmark.SplitTrain, manifest.TrainRecords},
		{manifest.ValidationURL, manifest.ValidationBlob, benchmark.SplitValidation, manifest.ValidationRecords},
		{manifest.TestURL, manifest.TestBlob, benchmark.SplitTest, manifest.TestRecords},
	}
	records := make([]benchmark.MultiLabelDatasetRecord, 0, manifest.TrainRecords+manifest.ValidationRecords+manifest.TestRecords)
	for _, source := range splits {
		content, err := fetchGitBlobContent(ctx, client, source.url, source.blob)
		if err != nil {
			return nil, fmt.Errorf("download %s split: %w", source.split, err)
		}
		parsed, err := parseGoEmotionsTSV(content, source.split, labels)
		if err != nil {
			return nil, fmt.Errorf("parse %s split: %w", source.split, err)
		}
		if len(parsed) != source.expected {
			return nil, fmt.Errorf("GoEmotions %s records = %d, want %d", source.split, len(parsed), source.expected)
		}
		records = append(records, parsed...)
	}
	return records, nil
}

func fetchGitBlobContent(ctx context.Context, client *http.Client, url, expectedBlob string) ([]byte, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create source request: %w", err)
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("download source: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download source returned HTTP %d", response.StatusCode)
	}
	content, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("read source response: %w", err)
	}
	if actualBlob := gitBlobSHA1(content); actualBlob != expectedBlob {
		return nil, fmt.Errorf("source blob checksum = %s, want %s", actualBlob, expectedBlob)
	}
	return content, nil
}

func parseGoEmotionsLabels(content []byte) ([]string, error) {
	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) == 0 {
		return nil, errors.New("labels file is empty")
	}
	labels := make([]string, 0, len(lines))
	seen := make(map[string]struct{}, len(lines))
	for line, label := range lines {
		label = strings.TrimSpace(label)
		if label == "" {
			return nil, fmt.Errorf("label line %d is empty", line+1)
		}
		if _, exists := seen[label]; exists {
			return nil, fmt.Errorf("duplicate label %q", label)
		}
		seen[label] = struct{}{}
		labels = append(labels, label)
	}
	return labels, nil
}

func parseGoEmotionsTSV(content []byte, split string, labels []string) ([]benchmark.MultiLabelDatasetRecord, error) {
	records := make([]benchmark.MultiLabelDatasetRecord, 0)
	scanner := bufio.NewScanner(bytes.NewReader(content))
	scanner.Buffer(make([]byte, 1024), 1024*1024)
	for line := 1; scanner.Scan(); line++ {
		row := strings.Split(scanner.Text(), "\t")
		if len(row) != 3 {
			return nil, fmt.Errorf("TSV line %d has %d fields, want 3", line, len(row))
		}
		labelIDs := strings.Split(row[1], ",")
		recordLabels := make([]string, 0, len(labelIDs))
		for _, encodedID := range labelIDs {
			labelID, err := strconv.Atoi(encodedID)
			if err != nil || labelID < 0 || labelID >= len(labels) {
				return nil, fmt.Errorf("TSV line %d has invalid label id %q", line, encodedID)
			}
			recordLabels = append(recordLabels, labels[labelID])
		}
		goldLabelsJSON, err := benchmark.EncodeGoldLabels(recordLabels)
		if err != nil {
			return nil, fmt.Errorf("TSV line %d: %w", line, err)
		}
		records = append(records, benchmark.MultiLabelDatasetRecord{
			ID:             fmt.Sprintf("goemotions-%s-%s", split, row[2]),
			Text:           row[0],
			Split:          split,
			GoldLabelsJSON: goldLabelsJSON,
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan TSV: %w", err)
	}
	return records, nil
}

func validateGoEmotionsManifest(manifest GoEmotionsManifest) error {
	if manifest.ID != "goemotions-simplified" {
		return fmt.Errorf("manifest id = %q, want goemotions-simplified", manifest.ID)
	}
	if manifest.License != "Apache-2.0" {
		return fmt.Errorf("manifest license = %q, want Apache-2.0", manifest.License)
	}
	if strings.TrimSpace(manifest.SourceRevision) == "" {
		return errors.New("manifest source revision must not be empty")
	}
	urls := []string{manifest.TrainURL, manifest.ValidationURL, manifest.TestURL, manifest.LabelsURL}
	for _, url := range urls {
		if strings.TrimSpace(url) == "" {
			return errors.New("manifest source URLs must not be empty")
		}
	}
	blobs := []string{manifest.TrainBlob, manifest.ValidationBlob, manifest.TestBlob, manifest.LabelsBlob}
	for _, blob := range blobs {
		if len(blob) != 40 {
			return errors.New("manifest source blob IDs must be SHA-1 values")
		}
	}
	if manifest.TrainRecords < 1 || manifest.ValidationRecords < 1 || manifest.TestRecords < 1 || manifest.LabelCount < 1 {
		return errors.New("manifest record and label counts must be positive")
	}
	return nil
}
