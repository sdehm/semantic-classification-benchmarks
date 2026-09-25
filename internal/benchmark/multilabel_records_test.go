package benchmark

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestMultiLabelDatasetRecordParquetRoundTrip(t *testing.T) {
	records := []MultiLabelDatasetRecord{
		{ID: "train-1", Text: "That is wonderful", Split: SplitTrain, GoldLabelsJSON: `["admiration","joy"]`},
		{ID: "test-1", Text: "I am afraid", Split: SplitTest, GoldLabelsJSON: `["fear"]`},
	}
	path := filepath.Join(t.TempDir(), "records.parquet")
	if err := WriteMultiLabelDatasetRecords(path, records); err != nil {
		t.Fatalf("WriteMultiLabelDatasetRecords() error = %v", err)
	}
	got, err := ReadMultiLabelDatasetRecords(path)
	if err != nil {
		t.Fatalf("ReadMultiLabelDatasetRecords() error = %v", err)
	}
	if !reflect.DeepEqual(got, records) {
		t.Fatalf("records = %#v, want %#v", got, records)
	}
}

func TestEncodeGoldLabelsCanonicalizesOrder(t *testing.T) {
	got, err := EncodeGoldLabels([]string{"joy", "admiration"})
	if err != nil {
		t.Fatalf("EncodeGoldLabels() error = %v", err)
	}
	if want := `["admiration","joy"]`; got != want {
		t.Fatalf("EncodeGoldLabels() = %q, want %q", got, want)
	}
}

func TestMultiLabelPredictionRecordParquetRoundTrip(t *testing.T) {
	records := []MultiLabelPredictionRecord{
		{ID: "test-1", PredictedLabelsJSON: `["joy"]`, LabelProbabilitiesJSON: `{"joy":0.9}`, Model: "test-model"},
	}
	path := filepath.Join(t.TempDir(), "predictions.parquet")
	if err := WriteMultiLabelPredictionRecords(path, records); err != nil {
		t.Fatalf("WriteMultiLabelPredictionRecords() error = %v", err)
	}
	got, err := ReadMultiLabelPredictionRecords(path)
	if err != nil {
		t.Fatalf("ReadMultiLabelPredictionRecords() error = %v", err)
	}
	if !reflect.DeepEqual(got, records) {
		t.Fatalf("records = %#v, want %#v", got, records)
	}
}
