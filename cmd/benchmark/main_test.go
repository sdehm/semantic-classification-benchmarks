package main

import (
	"strings"
	"testing"
)

func TestJevRunsRequireAccountInputPrice(t *testing.T) {
	for _, test := range []struct {
		name string
		run  func([]string) error
	}{
		{"choice", runJevRun},
		{"noul", runJevNoulRun},
	} {
		t.Run(test.name, func(t *testing.T) {
			err := test.run([]string{"--dry-run"})
			if err == nil || !strings.Contains(err.Error(), "--input-price-per-million") {
				t.Fatalf("run without a price error = %v", err)
			}
		})
	}
}
