package jev

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

func checkpointFingerprint(values ...any) (string, error) {
	digest := sha256.New()
	encoder := json.NewEncoder(digest)
	for _, value := range values {
		if err := encoder.Encode(value); err != nil {
			return "", fmt.Errorf("encode Jev checkpoint provenance: %w", err)
		}
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}
