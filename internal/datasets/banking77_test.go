package datasets

import (
	"testing"

	"github.com/sdehm/semantic-classification-benchmarks/internal/benchmark"
)

func TestParseBanking77CSV(t *testing.T) {
	records, err := parseBanking77CSV([]byte("text,category\n\"My card is late\",card_arrival\n"), benchmark.SplitTrain)
	if err != nil {
		t.Fatalf("parseBanking77CSV() error = %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("record count = %d, want 1", len(records))
	}
	if records[0].ID != "banking77-train-00000" || records[0].GoldLabel != "card_arrival" {
		t.Fatalf("record = %#v", records[0])
	}
}

func TestGitBlobSHA1(t *testing.T) {
	if got, want := gitBlobSHA1([]byte("hello\n")), "ce013625030ba8dba906f756967f9e9ca394464a"; got != want {
		t.Fatalf("gitBlobSHA1() = %q, want %q", got, want)
	}
}
