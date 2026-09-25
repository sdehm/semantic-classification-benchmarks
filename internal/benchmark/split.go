package benchmark

import (
	"crypto/sha256"
	"fmt"
	"math"
	"sort"
)

// AssignValidationSplit replaces only train records with a deterministic,
// stratified validation split. Test records are never altered.
func AssignValidationSplit(records []DatasetRecord, fraction float64, seed string) ([]DatasetRecord, error) {
	if err := ValidateDatasetRecords(records); err != nil {
		return nil, err
	}
	if fraction <= 0 || fraction >= 1 {
		return nil, fmt.Errorf("validation fraction must be between 0 and 1, got %f", fraction)
	}
	if seed == "" {
		return nil, fmt.Errorf("split seed must not be empty")
	}

	result := append([]DatasetRecord(nil), records...)
	byLabel := make(map[string][]rankedRecord)
	for index, record := range result {
		if record.Split != SplitTrain {
			continue
		}
		hash := sha256.Sum256([]byte(seed + "\x00" + record.GoldLabel + "\x00" + record.ID))
		byLabel[record.GoldLabel] = append(byLabel[record.GoldLabel], rankedRecord{index: index, hash: hash})
	}

	for label, group := range byLabel {
		sort.Slice(group, func(left, right int) bool {
			return string(group[left].hash[:]) < string(group[right].hash[:])
		})

		validationCount := int(math.Round(float64(len(group)) * fraction))
		if len(group) > 1 {
			validationCount = max(1, min(validationCount, len(group)-1))
		}
		for _, ranked := range group[:validationCount] {
			result[ranked.index].Split = SplitValidation
		}

		if validationCount == 0 {
			return nil, fmt.Errorf("label %q cannot produce a validation record", label)
		}
	}
	return result, nil
}

type rankedRecord struct {
	index int
	hash  [sha256.Size]byte
}
