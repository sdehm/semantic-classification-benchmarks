package datasets

import (
	"strings"
	"testing"

	"github.com/sdehm/semantic-classification-benchmarks/internal/benchmark"
)

func TestParseMultiEURLEXJSONL(t *testing.T) {
	records, sourceRecords, err := parseMultiEURLEXJSONL(
		strings.NewReader(
			"{\"celex_id\":\"31979D0509\",\"text\":{\"en\":\"A legal document\"},\"eurovoc_concepts\":{\"level_1\":[\"100149\",\"100160\"]}}\n"+
				"{\"celex_id\":\"32006D0213\",\"text\":{\"de\":\"Ein Rechtsakt\"},\"eurovoc_concepts\":{\"level_1\":[\"100149\"]}}\n",
		),
		benchmark.SplitTrain,
		"level_1",
		map[string]string{
			"100149": "100149: social questions",
			"100160": "100160: international affairs",
		},
	)
	if err != nil {
		t.Fatalf("parseMultiEURLEXJSONL() error = %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("record count = %d, want 1", len(records))
	}
	if sourceRecords != 2 {
		t.Fatalf("source record count = %d, want 2", sourceRecords)
	}
	if records[0].ID != "multieurlex-level1-train-31979D0509" {
		t.Fatalf("record ID = %q", records[0].ID)
	}
	if records[0].GoldLabelsJSON != `["100149: social questions","100160: international affairs"]` {
		t.Fatalf("labels = %q", records[0].GoldLabelsJSON)
	}
}

func TestParseMultiEURLEXDescriptorsRejectsMissingEnglishText(t *testing.T) {
	_, err := parseMultiEURLEXDescriptors(
		[]byte(`{"100149":{"fr":"questions sociales"}}`),
		[]string{"100149"},
	)
	if err == nil {
		t.Fatal("parseMultiEURLEXDescriptors() returned nil error")
	}
}
