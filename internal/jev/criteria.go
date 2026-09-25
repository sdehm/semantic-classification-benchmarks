package jev

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"

	"github.com/sdehm/jev-classify-test/internal/benchmark"
)

// EnrichCriteriaWithTrainingExamples appends deterministic, label-matched
// examples from the training split. It never reads validation or test text.
func EnrichCriteriaWithTrainingExamples(
	records []benchmark.DatasetRecord,
	criteria map[string]string,
	examplesPerLabel int,
	seed string,
) (map[string]string, error) {
	if examplesPerLabel < 0 {
		return nil, fmt.Errorf("examples per label must not be negative, got %d", examplesPerLabel)
	}
	if examplesPerLabel == 0 {
		return cloneCriteria(criteria), nil
	}
	if seed == "" {
		return nil, fmt.Errorf("training-example seed must not be empty")
	}

	type rankedExample struct {
		text string
		hash [sha256.Size]byte
	}
	examplesByLabel := make(map[string][]rankedExample)
	for _, record := range records {
		if record.Split != benchmark.SplitTrain {
			continue
		}
		hash := sha256.Sum256([]byte(seed + "\x00" + record.GoldLabel + "\x00" + record.ID))
		examplesByLabel[record.GoldLabel] = append(examplesByLabel[record.GoldLabel], rankedExample{
			text: record.Text,
			hash: hash,
		})
	}

	enriched := cloneCriteria(criteria)
	for label, description := range enriched {
		examples := examplesByLabel[label]
		if len(examples) < examplesPerLabel {
			return nil, fmt.Errorf(
				"label %q has %d training examples, need %d",
				label,
				len(examples),
				examplesPerLabel,
			)
		}
		sort.Slice(examples, func(left, right int) bool {
			return string(examples[left].hash[:]) < string(examples[right].hash[:])
		})
		enriched[label] = description + "\n\nRepresentative training examples:"
		for _, example := range examples[:examplesPerLabel] {
			enriched[label] += "\n- " + strconv.Quote(example.text)
		}
	}
	return enriched, nil
}

func CriteriaHash(criteria map[string]string) (string, error) {
	encoded, err := json.Marshal(criteria)
	if err != nil {
		return "", fmt.Errorf("encode criteria for hashing: %w", err)
	}
	hash := sha256.Sum256(encoded)
	return hex.EncodeToString(hash[:]), nil
}

func cloneCriteria(criteria map[string]string) map[string]string {
	result := make(map[string]string, len(criteria))
	for label, description := range criteria {
		result[label] = description
	}
	return result
}
