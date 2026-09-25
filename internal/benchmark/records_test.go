package benchmark

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestDatasetRecordParquetRoundTrip(t *testing.T) {
	records := []DatasetRecord{
		{ID: "train-1", Text: "My card is late", Split: SplitTrain, GoldLabel: "card_arrival"},
		{ID: "test-1", Text: "A transfer failed", Split: SplitTest, GoldLabel: "failed_transfer"},
	}
	path := filepath.Join(t.TempDir(), "records.parquet")
	if err := WriteDatasetRecords(path, records); err != nil {
		t.Fatalf("WriteDatasetRecords() error = %v", err)
	}
	got, err := ReadDatasetRecords(path)
	if err != nil {
		t.Fatalf("ReadDatasetRecords() error = %v", err)
	}
	if !reflect.DeepEqual(got, records) {
		t.Fatalf("records = %#v, want %#v", got, records)
	}
}

func TestAssignValidationSplitPreservesTestAndLabels(t *testing.T) {
	records := []DatasetRecord{
		{ID: "a-1", Text: "a", Split: SplitTrain, GoldLabel: "a"},
		{ID: "a-2", Text: "a", Split: SplitTrain, GoldLabel: "a"},
		{ID: "b-1", Text: "b", Split: SplitTrain, GoldLabel: "b"},
		{ID: "b-2", Text: "b", Split: SplitTrain, GoldLabel: "b"},
		{ID: "test-1", Text: "test", Split: SplitTest, GoldLabel: "a"},
	}
	got, err := AssignValidationSplit(records, 0.5, "seed-1")
	if err != nil {
		t.Fatalf("AssignValidationSplit() error = %v", err)
	}
	again, err := AssignValidationSplit(records, 0.5, "seed-1")
	if err != nil {
		t.Fatalf("AssignValidationSplit() repeat error = %v", err)
	}
	if !reflect.DeepEqual(got, again) {
		t.Fatal("split is not deterministic")
	}
	if got[4].Split != SplitTest {
		t.Fatalf("test split changed to %q", got[4].Split)
	}
	validationByLabel := map[string]int{}
	for _, record := range got {
		if record.Split == SplitValidation {
			validationByLabel[record.GoldLabel]++
		}
	}
	if validationByLabel["a"] != 1 || validationByLabel["b"] != 1 {
		t.Fatalf("validation labels = %#v, want one record per label", validationByLabel)
	}
}
