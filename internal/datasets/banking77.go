package datasets

import (
	"bytes"
	"context"
	"crypto/sha1"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/sdehm/jev-classify-test/internal/benchmark"
)

type Banking77Manifest struct {
	ID                 string  `json:"id"`
	License            string  `json:"license"`
	SourceRepository   string  `json:"source_repository"`
	SourceRevision     string  `json:"source_revision"`
	TrainURL           string  `json:"train_url"`
	TrainBlob          string  `json:"train_blob"`
	TrainRecords       int     `json:"train_records"`
	TestURL            string  `json:"test_url"`
	TestBlob           string  `json:"test_blob"`
	TestRecords        int     `json:"test_records"`
	ValidationSeed     string  `json:"validation_seed"`
	ValidationFraction float64 `json:"validation_fraction"`
	Citation           string  `json:"citation"`
}

func LoadBanking77Manifest(path string) (Banking77Manifest, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return Banking77Manifest{}, fmt.Errorf("read BANKING77 manifest: %w", err)
	}
	var manifest Banking77Manifest
	if err := json.Unmarshal(content, &manifest); err != nil {
		return Banking77Manifest{}, fmt.Errorf("decode BANKING77 manifest: %w", err)
	}
	if err := validateBanking77Manifest(manifest); err != nil {
		return Banking77Manifest{}, err
	}
	return manifest, nil
}

// PrepareBanking77 downloads and verifies the pinned source files, then creates
// records for the common benchmark contract. It does not write raw data to disk.
func PrepareBanking77(ctx context.Context, client *http.Client, manifest Banking77Manifest, validationFraction float64, seed string) ([]benchmark.DatasetRecord, error) {
	if err := validateBanking77Manifest(manifest); err != nil {
		return nil, err
	}
	if client == nil {
		client = http.DefaultClient
	}

	train, err := fetchAndParseBanking77(ctx, client, manifest.TrainURL, manifest.TrainBlob, benchmark.SplitTrain)
	if err != nil {
		return nil, fmt.Errorf("prepare BANKING77 train records: %w", err)
	}
	if len(train) != manifest.TrainRecords {
		return nil, fmt.Errorf("BANKING77 train record count = %d, want %d", len(train), manifest.TrainRecords)
	}
	test, err := fetchAndParseBanking77(ctx, client, manifest.TestURL, manifest.TestBlob, benchmark.SplitTest)
	if err != nil {
		return nil, fmt.Errorf("prepare BANKING77 test records: %w", err)
	}
	if len(test) != manifest.TestRecords {
		return nil, fmt.Errorf("BANKING77 test record count = %d, want %d", len(test), manifest.TestRecords)
	}

	records := append(train, test...)
	return benchmark.AssignValidationSplit(records, validationFraction, seed)
}

func fetchAndParseBanking77(ctx context.Context, client *http.Client, url, expectedBlob, split string) ([]benchmark.DatasetRecord, error) {
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
	return parseBanking77CSV(content, split)
}

func parseBanking77CSV(content []byte, split string) ([]benchmark.DatasetRecord, error) {
	reader := csv.NewReader(bytes.NewReader(content))
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("read CSV header: %w", err)
	}
	if len(header) != 2 || header[0] != "text" || header[1] != "category" {
		return nil, fmt.Errorf("unexpected CSV header %q", header)
	}

	records := make([]benchmark.DatasetRecord, 0)
	for line := 2; ; line++ {
		row, err := reader.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read CSV line %d: %w", line, err)
		}
		if len(row) != 2 {
			return nil, fmt.Errorf("CSV line %d has %d fields, want 2", line, len(row))
		}
		records = append(records, benchmark.DatasetRecord{
			ID:        fmt.Sprintf("banking77-%s-%05d", split, len(records)),
			Text:      row[0],
			Split:     split,
			GoldLabel: row[1],
		})
	}
	return records, nil
}

func gitBlobSHA1(content []byte) string {
	hash := sha1.New() // Git's blob object format defines SHA-1 for this source revision.
	_, _ = fmt.Fprintf(hash, "blob %d\x00", len(content))
	_, _ = hash.Write(content)
	return hex.EncodeToString(hash.Sum(nil))
}

func validateBanking77Manifest(manifest Banking77Manifest) error {
	if manifest.ID != "banking77" {
		return fmt.Errorf("manifest id = %q, want %q", manifest.ID, "banking77")
	}
	if manifest.License != "CC-BY-4.0" {
		return fmt.Errorf("manifest license = %q, want CC-BY-4.0", manifest.License)
	}
	if strings.TrimSpace(manifest.SourceRevision) == "" {
		return errors.New("manifest source revision must not be empty")
	}
	if strings.TrimSpace(manifest.TrainURL) == "" || strings.TrimSpace(manifest.TestURL) == "" {
		return errors.New("manifest source URLs must not be empty")
	}
	if len(manifest.TrainBlob) != 40 || len(manifest.TestBlob) != 40 {
		return errors.New("manifest source blob IDs must be SHA-1 values")
	}
	if manifest.TrainRecords < 1 || manifest.TestRecords < 1 {
		return errors.New("manifest source record counts must be positive")
	}
	if strings.TrimSpace(manifest.ValidationSeed) == "" {
		return errors.New("manifest validation seed must not be empty")
	}
	if manifest.ValidationFraction <= 0 || manifest.ValidationFraction >= 1 {
		return fmt.Errorf("manifest validation fraction must be between 0 and 1, got %f", manifest.ValidationFraction)
	}
	return nil
}
