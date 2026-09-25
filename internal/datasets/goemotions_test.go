package datasets

import (
	"testing"

	"github.com/sdehm/semantic-classification-benchmarks/internal/benchmark"
)

func TestParseGoEmotionsTSV(t *testing.T) {
	labels, err := parseGoEmotionsLabels([]byte("admiration\njoy\nneutral"))
	if err != nil {
		t.Fatalf("parseGoEmotionsLabels() error = %v", err)
	}
	records, err := parseGoEmotionsTSV(
		[]byte("Wonderful news\t0,1\tcomment-1\nNothing much\t2\tcomment-2\n"),
		benchmark.SplitTrain,
		labels,
	)
	if err != nil {
		t.Fatalf("parseGoEmotionsTSV() error = %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("record count = %d, want 2", len(records))
	}
	if records[0].ID != "goemotions-train-comment-1" || records[0].GoldLabelsJSON != `["admiration","joy"]` {
		t.Fatalf("first record = %#v", records[0])
	}
	if records[1].GoldLabelsJSON != `["neutral"]` {
		t.Fatalf("second record = %#v", records[1])
	}
}
