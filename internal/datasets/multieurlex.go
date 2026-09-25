package datasets

import (
	"archive/zip"
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/sdehm/semantic-classification-benchmarks/internal/benchmark"
)

type MultiEURLEXManifest struct {
	ID                         string   `json:"id"`
	DatasetCardMetadataLicense string   `json:"dataset_card_metadata_license"`
	DatasetCardTextLicense     string   `json:"dataset_card_text_license"`
	DatasetRepository          string   `json:"dataset_repository"`
	DatasetRevision            string   `json:"dataset_revision"`
	ArchiveURL                 string   `json:"archive_url"`
	ArchiveSHA256              string   `json:"archive_sha256"`
	ArchiveBytes               int64    `json:"archive_bytes"`
	SourceRepository           string   `json:"source_repository"`
	SourceRevision             string   `json:"source_revision"`
	DescriptorsURL             string   `json:"descriptors_url"`
	DescriptorsBlob            string   `json:"descriptors_blob"`
	LabelLevel                 string   `json:"label_level"`
	LabelIDs                   []string `json:"label_ids"`
	SourceTrainRecords         int      `json:"source_train_records"`
	SourceValidationRecords    int      `json:"source_validation_records"`
	SourceTestRecords          int      `json:"source_test_records"`
	TrainRecords               int      `json:"train_records"`
	ValidationRecords          int      `json:"validation_records"`
	TestRecords                int      `json:"test_records"`
	Citation                   string   `json:"citation"`
	OriginalDatasetCitation    string   `json:"original_dataset_citation"`
}

const multiEURLEXMaximumLineBytes = 128 * 1024 * 1024

func LoadMultiEURLEXManifest(path string) (MultiEURLEXManifest, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return MultiEURLEXManifest{}, fmt.Errorf("read MultiEURLEX manifest: %w", err)
	}
	var manifest MultiEURLEXManifest
	if err := json.Unmarshal(content, &manifest); err != nil {
		return MultiEURLEXManifest{}, fmt.Errorf("decode MultiEURLEX manifest: %w", err)
	}
	if err := validateMultiEURLEXManifest(manifest); err != nil {
		return MultiEURLEXManifest{}, err
	}
	return manifest, nil
}

// PrepareMultiEURLEX verifies a cached pinned archive and returns English
// level-1 records. The archive and normalized Parquet output remain local.
func PrepareMultiEURLEX(
	ctx context.Context,
	client *http.Client,
	manifest MultiEURLEXManifest,
	archivePath string,
) ([]benchmark.MultiLabelDatasetRecord, error) {
	if err := validateMultiEURLEXManifest(manifest); err != nil {
		return nil, err
	}
	if client == nil {
		client = http.DefaultClient
	}
	if err := downloadMultiEURLEXArchive(ctx, client, manifest, archivePath); err != nil {
		return nil, err
	}
	descriptorContent, err := fetchGitBlobContent(
		ctx, client, manifest.DescriptorsURL, manifest.DescriptorsBlob,
	)
	if err != nil {
		return nil, fmt.Errorf("download EUROVOC descriptors: %w", err)
	}
	descriptors, err := parseMultiEURLEXDescriptors(descriptorContent, manifest.LabelIDs)
	if err != nil {
		return nil, err
	}

	archive, err := zip.OpenReader(archivePath)
	if err != nil {
		return nil, fmt.Errorf("open MultiEURLEX archive: %w", err)
	}
	defer archive.Close()

	type sourceSplit struct {
		filename       string
		split          string
		sourceExpected int
		expected       int
	}
	splits := []sourceSplit{
		{
			filename:       "train.jsonl",
			split:          benchmark.SplitTrain,
			sourceExpected: manifest.SourceTrainRecords,
			expected:       manifest.TrainRecords,
		},
		{
			filename:       "dev.jsonl",
			split:          benchmark.SplitValidation,
			sourceExpected: manifest.SourceValidationRecords,
			expected:       manifest.ValidationRecords,
		},
		{
			filename:       "test.jsonl",
			split:          benchmark.SplitTest,
			sourceExpected: manifest.SourceTestRecords,
			expected:       manifest.TestRecords,
		},
	}
	records := make([]benchmark.MultiLabelDatasetRecord, 0, manifest.TrainRecords+manifest.ValidationRecords+manifest.TestRecords)
	for _, source := range splits {
		file, err := multiEURLEXArchiveFile(archive.File, source.filename)
		if err != nil {
			return nil, err
		}
		handle, err := file.Open()
		if err != nil {
			return nil, fmt.Errorf("open %s in MultiEURLEX archive: %w", source.filename, err)
		}
		parsed, sourceRecords, parseErr := parseMultiEURLEXJSONL(
			handle, source.split, manifest.LabelLevel, descriptors,
		)
		closeErr := handle.Close()
		if parseErr != nil || closeErr != nil {
			return nil, fmt.Errorf("parse %s: %w", source.filename, errors.Join(parseErr, closeErr))
		}
		if sourceRecords != source.sourceExpected {
			return nil, fmt.Errorf(
				"MultiEURLEX %s source records = %d, want %d",
				source.split, sourceRecords, source.sourceExpected,
			)
		}
		if len(parsed) != source.expected {
			return nil, fmt.Errorf("MultiEURLEX %s records = %d, want %d", source.split, len(parsed), source.expected)
		}
		records = append(records, parsed...)
	}
	observedLabels := make(map[string]struct{}, len(descriptors))
	for _, record := range records {
		labels, err := benchmark.DecodeGoldLabels(record.GoldLabelsJSON)
		if err != nil {
			return nil, fmt.Errorf("decode prepared MultiEURLEX labels for %q: %w", record.ID, err)
		}
		for _, label := range labels {
			observedLabels[label] = struct{}{}
		}
	}
	if len(observedLabels) != len(descriptors) {
		return nil, fmt.Errorf(
			"prepared MultiEURLEX covers %d level-1 labels, want %d",
			len(observedLabels), len(descriptors),
		)
	}
	return records, nil
}

func downloadMultiEURLEXArchive(
	ctx context.Context,
	client *http.Client,
	manifest MultiEURLEXManifest,
	archivePath string,
) error {
	if info, err := os.Stat(archivePath); err == nil {
		if info.Size() == manifest.ArchiveBytes {
			actual, hashErr := fileSHA256(archivePath)
			if hashErr == nil && actual == manifest.ArchiveSHA256 {
				return nil
			}
		}
		if err := os.Remove(archivePath); err != nil {
			return fmt.Errorf("remove invalid MultiEURLEX archive: %w", err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("stat MultiEURLEX archive: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(archivePath), 0o755); err != nil {
		return fmt.Errorf("create MultiEURLEX archive directory: %w", err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, manifest.ArchiveURL, nil)
	if err != nil {
		return fmt.Errorf("create MultiEURLEX archive request: %w", err)
	}
	response, err := client.Do(request)
	if err != nil {
		return fmt.Errorf("download MultiEURLEX archive: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("MultiEURLEX archive returned HTTP %d", response.StatusCode)
	}
	temporaryPath := archivePath + ".partial"
	file, err := os.Create(temporaryPath)
	if err != nil {
		return fmt.Errorf("create temporary MultiEURLEX archive: %w", err)
	}
	hasher := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(file, hasher), response.Body)
	closeErr := file.Close()
	if copyErr != nil || closeErr != nil {
		_ = os.Remove(temporaryPath)
		return fmt.Errorf("write MultiEURLEX archive: %w", errors.Join(copyErr, closeErr))
	}
	actual := hex.EncodeToString(hasher.Sum(nil))
	if written != manifest.ArchiveBytes || actual != manifest.ArchiveSHA256 {
		_ = os.Remove(temporaryPath)
		return fmt.Errorf(
			"MultiEURLEX archive verification failed: size %d (want %d), SHA-256 %s (want %s)",
			written, manifest.ArchiveBytes, actual, manifest.ArchiveSHA256,
		)
	}
	if err := os.Rename(temporaryPath, archivePath); err != nil {
		return fmt.Errorf("move verified MultiEURLEX archive into place: %w", err)
	}
	return nil
}

func fileSHA256(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func multiEURLEXArchiveFile(files []*zip.File, filename string) (*zip.File, error) {
	for _, file := range files {
		if path.Base(file.Name) == filename {
			return file, nil
		}
	}
	return nil, fmt.Errorf("MultiEURLEX archive does not contain %q", filename)
}

func parseMultiEURLEXDescriptors(content []byte, labelIDs []string) (map[string]string, error) {
	var source map[string]map[string]string
	if err := json.Unmarshal(content, &source); err != nil {
		return nil, fmt.Errorf("decode EUROVOC descriptors: %w", err)
	}
	descriptors := make(map[string]string, len(labelIDs))
	for _, id := range labelIDs {
		descriptor := strings.TrimSpace(source[id]["en"])
		if descriptor == "" {
			return nil, fmt.Errorf("EUROVOC descriptor %q has no English text", id)
		}
		descriptors[id] = fmt.Sprintf("%s: %s", id, descriptor)
	}
	return descriptors, nil
}

func parseMultiEURLEXJSONL(
	reader io.Reader,
	split, labelLevel string,
	descriptors map[string]string,
) ([]benchmark.MultiLabelDatasetRecord, int, error) {
	type sourceRecord struct {
		CELEXID         string              `json:"celex_id"`
		Text            map[string]*string  `json:"text"`
		EurovocConcepts map[string][]string `json:"eurovoc_concepts"`
	}
	records := make([]benchmark.MultiLabelDatasetRecord, 0)
	sourceRecords := 0
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), multiEURLEXMaximumLineBytes)
	for line := 1; scanner.Scan(); line++ {
		sourceRecords++
		var source sourceRecord
		if err := json.Unmarshal(scanner.Bytes(), &source); err != nil {
			return nil, 0, fmt.Errorf("JSONL line %d: %w", line, err)
		}
		if source.Text["en"] == nil || strings.TrimSpace(*source.Text["en"]) == "" {
			continue
		}
		if strings.TrimSpace(source.CELEXID) == "" {
			return nil, 0, fmt.Errorf("JSONL line %d is missing a CELEX ID", line)
		}
		labelIDs, exists := source.EurovocConcepts[labelLevel]
		if !exists || len(labelIDs) == 0 {
			return nil, 0, fmt.Errorf("JSONL line %d has no %s labels", line, labelLevel)
		}
		labels := make([]string, 0, len(labelIDs))
		for _, id := range labelIDs {
			label, exists := descriptors[id]
			if !exists {
				return nil, 0, fmt.Errorf("JSONL line %d has unsupported label ID %q", line, id)
			}
			labels = append(labels, label)
		}
		goldLabelsJSON, err := benchmark.EncodeGoldLabels(labels)
		if err != nil {
			return nil, 0, fmt.Errorf("JSONL line %d: %w", line, err)
		}
		records = append(records, benchmark.MultiLabelDatasetRecord{
			ID:             fmt.Sprintf("multieurlex-level1-%s-%s", split, source.CELEXID),
			Text:           *source.Text["en"],
			Split:          split,
			GoldLabelsJSON: goldLabelsJSON,
		})
	}
	if err := scanner.Err(); err != nil {
		return nil, 0, fmt.Errorf("scan MultiEURLEX JSONL: %w", err)
	}
	return records, sourceRecords, nil
}

func validateMultiEURLEXManifest(manifest MultiEURLEXManifest) error {
	if manifest.ID != "multieurlex-english-level1" {
		return fmt.Errorf("manifest id = %q, want multieurlex-english-level1", manifest.ID)
	}
	if manifest.DatasetCardMetadataLicense != "CC-BY-SA-4.0" ||
		manifest.DatasetCardTextLicense != "CC-BY-4.0" {
		return fmt.Errorf(
			"MultiEURLEX dataset card license claims = %q (metadata), %q (text), want CC-BY-SA-4.0 and CC-BY-4.0",
			manifest.DatasetCardMetadataLicense, manifest.DatasetCardTextLicense,
		)
	}
	if strings.TrimSpace(manifest.DatasetRevision) == "" || strings.TrimSpace(manifest.SourceRevision) == "" {
		return errors.New("dataset and source revisions must not be empty")
	}
	if strings.TrimSpace(manifest.ArchiveURL) == "" || strings.TrimSpace(manifest.DescriptorsURL) == "" {
		return errors.New("archive and descriptor URLs must not be empty")
	}
	if len(manifest.ArchiveSHA256) != sha256.Size*2 {
		return errors.New("archive SHA-256 must be a 64-character hexadecimal value")
	}
	if manifest.ArchiveBytes < 1 {
		return errors.New("archive size must be positive")
	}
	if len(manifest.DescriptorsBlob) != 40 {
		return errors.New("descriptor blob must be a Git SHA-1 value")
	}
	if manifest.LabelLevel != "level_1" || len(manifest.LabelIDs) != 21 {
		return errors.New("manifest must define the 21 level_1 label IDs")
	}
	seen := make(map[string]struct{}, len(manifest.LabelIDs))
	for _, id := range manifest.LabelIDs {
		if strings.TrimSpace(id) == "" {
			return errors.New("label IDs must not be empty")
		}
		if _, exists := seen[id]; exists {
			return fmt.Errorf("duplicate label ID %q", id)
		}
		seen[id] = struct{}{}
	}
	if manifest.SourceTrainRecords < manifest.TrainRecords ||
		manifest.SourceValidationRecords < manifest.ValidationRecords ||
		manifest.SourceTestRecords < manifest.TestRecords {
		return errors.New("source record counts must be at least the English record counts")
	}
	if manifest.TrainRecords < 1 || manifest.ValidationRecords < 1 || manifest.TestRecords < 1 {
		return errors.New("record counts must be positive")
	}
	return nil
}
