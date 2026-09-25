package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/sdehm/jev-classify-test/internal/benchmark"
	"github.com/sdehm/jev-classify-test/internal/datasets"
	"github.com/sdehm/jev-classify-test/internal/jev"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	var err error
	switch os.Args[1] {
	case "prepare-banking77":
		err = runPrepareBanking77(os.Args[2:])
	case "prepare-goemotions":
		err = runPrepareGoEmotions(os.Args[2:])
	case "prepare-multieurlex":
		err = runPrepareMultiEURLEX(os.Args[2:])
	case "split":
		err = runSplit(os.Args[2:])
	case "metrics":
		err = runMetrics(os.Args[2:])
	case "metrics-multilabel":
		err = runMultiLabelMetrics(os.Args[2:])
	case "jev-choice":
		err = runJevChoice(os.Args[2:])
	case "jev-run":
		err = runJevRun(os.Args[2:])
	case "jev-noul-run":
		err = runJevNoulRun(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func runPrepareBanking77(arguments []string) error {
	flags := flag.NewFlagSet("prepare-banking77", flag.ContinueOnError)
	manifestPath := flags.String("manifest", "configs/datasets/banking77.json", "pinned BANKING77 manifest")
	output := flags.String("output", "data/banking77/records.parquet", "output dataset Parquet path")
	seed := flags.String("seed", "", "override the manifest's deterministic split seed")
	fraction := flags.Float64("validation-fraction", -1, "override the manifest's validation fraction")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	manifest, err := datasets.LoadBanking77Manifest(*manifestPath)
	if err != nil {
		return err
	}
	if *seed == "" {
		*seed = manifest.ValidationSeed
	}
	if *fraction < 0 {
		*fraction = manifest.ValidationFraction
	}
	records, err := datasets.PrepareBanking77(context.Background(), nil, manifest, *fraction, *seed)
	if err != nil {
		return err
	}
	return benchmark.WriteDatasetRecords(*output, records)
}

func runPrepareGoEmotions(arguments []string) error {
	flags := flag.NewFlagSet("prepare-goemotions", flag.ContinueOnError)
	manifestPath := flags.String("manifest", "configs/datasets/goemotions.json", "pinned GoEmotions manifest")
	output := flags.String("output", "data/goemotions/records.parquet", "output multi-label dataset Parquet path")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	manifest, err := datasets.LoadGoEmotionsManifest(*manifestPath)
	if err != nil {
		return err
	}
	records, err := datasets.PrepareGoEmotions(context.Background(), nil, manifest)
	if err != nil {
		return err
	}
	return benchmark.WriteMultiLabelDatasetRecords(*output, records)
}

func runPrepareMultiEURLEX(arguments []string) error {
	flags := flag.NewFlagSet("prepare-multieurlex", flag.ContinueOnError)
	manifestPath := flags.String(
		"manifest",
		"configs/datasets/multieurlex-english-level1.json",
		"pinned English MultiEURLEX level-1 manifest",
	)
	archivePath := flags.String(
		"archive",
		"data/multieurlex/multi_eurlex_translated.zip",
		"verified source archive path",
	)
	output := flags.String(
		"output",
		"data/multieurlex/records.parquet",
		"output multi-label dataset Parquet path",
	)
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	manifest, err := datasets.LoadMultiEURLEXManifest(*manifestPath)
	if err != nil {
		return err
	}
	records, err := datasets.PrepareMultiEURLEX(context.Background(), nil, manifest, *archivePath)
	if err != nil {
		return err
	}
	return benchmark.WriteMultiLabelDatasetRecords(*output, records)
}

func runSplit(arguments []string) error {
	flags := flag.NewFlagSet("split", flag.ContinueOnError)
	input := flags.String("input", "", "input dataset Parquet path")
	output := flags.String("output", "", "output dataset Parquet path")
	seed := flags.String("seed", "", "non-empty deterministic split seed")
	fraction := flags.Float64("validation-fraction", 0.1, "fraction of each train label assigned to validation")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *input == "" || *output == "" {
		return fmt.Errorf("input and output are required")
	}
	records, err := benchmark.ReadDatasetRecords(*input)
	if err != nil {
		return err
	}
	records, err = benchmark.AssignValidationSplit(records, *fraction, *seed)
	if err != nil {
		return err
	}
	return benchmark.WriteDatasetRecords(*output, records)
}

func runMetrics(arguments []string) error {
	flags := flag.NewFlagSet("metrics", flag.ContinueOnError)
	dataset := flags.String("dataset", "", "dataset Parquet path")
	predictions := flags.String("predictions", "", "prediction Parquet path")
	split := flags.String("split", benchmark.SplitTest, "dataset split to evaluate")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *dataset == "" || *predictions == "" {
		return fmt.Errorf("dataset and predictions are required")
	}
	records, err := benchmark.ReadDatasetRecords(*dataset)
	if err != nil {
		return err
	}
	predictionRecords, err := benchmark.ReadPredictionRecords(*predictions)
	if err != nil {
		return err
	}
	report, err := benchmark.EvaluateSingleLabel(records, predictionRecords, *split)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(report)
}

func runMultiLabelMetrics(arguments []string) error {
	flags := flag.NewFlagSet("metrics-multilabel", flag.ContinueOnError)
	dataset := flags.String("dataset", "", "multi-label dataset Parquet path")
	predictions := flags.String("predictions", "", "multi-label prediction Parquet path")
	split := flags.String("split", benchmark.SplitTest, "dataset split to evaluate")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *dataset == "" || *predictions == "" {
		return fmt.Errorf("dataset and predictions are required")
	}
	records, err := benchmark.ReadMultiLabelDatasetRecords(*dataset)
	if err != nil {
		return err
	}
	predictionRecords, err := benchmark.ReadMultiLabelPredictionRecords(*predictions)
	if err != nil {
		return err
	}
	report, err := benchmark.EvaluateMultiLabel(records, predictionRecords, *split)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(report)
}

func runJevChoice(arguments []string) error {
	flags := flag.NewFlagSet("jev-choice", flag.ContinueOnError)
	text := flags.String("text", "", "text to classify")
	criteriaPath := flags.String(
		"criteria",
		"configs/datasets/banking77-label-criteria.json",
		"path to a JSON object mapping labels to criteria",
	)
	model := flags.String("model", "jev-latest", "TypeSafe model identifier")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if *text == "" {
		return fmt.Errorf("text is required")
	}
	criteriaBytes, err := os.ReadFile(*criteriaPath)
	if err != nil {
		return fmt.Errorf("read criteria: %w", err)
	}
	var criteria map[string]string
	if err := json.Unmarshal(criteriaBytes, &criteria); err != nil {
		return fmt.Errorf("decode criteria JSON: %w", err)
	}
	if len(criteria) == 0 {
		return fmt.Errorf("criteria JSON must contain at least one label")
	}
	for label, criterion := range criteria {
		if label == "" || criterion == "" {
			return fmt.Errorf("criteria JSON contains an empty label or description")
		}
	}
	client, err := jev.NewClient(jev.ClientConfig{APIKey: os.Getenv("TYPESAFE_API_KEY")})
	if err != nil {
		return err
	}
	started := time.Now()
	response, err := client.Evaluate(context.Background(), jev.SystemOneRequest{
		State: *text,
		Model: *model,
		Questions: map[string]jev.Question{
			"label": {
				Type:         "choice",
				Instructions: "Classify this text using exactly one label from the criteria.",
				Criteria:     criteria,
			},
		},
	})
	if err != nil {
		return err
	}
	result := struct {
		Response      jev.SystemOneResponse `json:"response"`
		LatencyMillis int64                 `json:"latency_ms"`
	}{Response: response, LatencyMillis: time.Since(started).Milliseconds()}
	return json.NewEncoder(os.Stdout).Encode(result)
}

func runJevRun(arguments []string) error {
	flags := flag.NewFlagSet("jev-run", flag.ContinueOnError)
	datasetPath := flags.String("dataset", "data/banking77/records.parquet", "input dataset Parquet path")
	criteriaPath := flags.String(
		"criteria",
		"configs/datasets/banking77-label-criteria.json",
		"path to a JSON object mapping labels to criteria",
	)
	outputPath := flags.String("output", "runs/jev/validation-v2.predictions.parquet", "prediction Parquet path")
	checkpointPath := flags.String("checkpoint", "runs/jev/validation-v2.checkpoint.jsonl", "durable JSONL checkpoint path")
	model := flags.String("model", "jev-1.13.0", "pinned TypeSafe model identifier")
	split := flags.String("split", benchmark.SplitValidation, "dataset split to classify")
	limit := flags.Int("limit", 25, "maximum records to include in this resumable run")
	sampleSeed := flags.String("sample-seed", "jev-validation-sample-v1", "deterministic seed for a limited validation sample")
	examplesPerLabel := flags.Int("examples-per-label", 0, "training examples appended to each label criterion")
	exampleSeed := flags.String("example-seed", "jev-training-examples-v1", "deterministic seed for training examples")
	maxCost := flags.Float64("max-cost-usd", 0.50, "local hard ceiling in US dollars")
	price := flags.Float64("input-price-per-million", 0, "your account's price per million input tokens (required)")
	maxTokens := flags.Int("max-input-tokens-per-request", jev.MaxInputTokensPerRequest, "per-request token reservation")
	dryRun := flags.Bool("dry-run", false, "show the safe request plan without calling Jev")
	allowLegacy := flags.Bool("allow-legacy-checkpoint-replay", false, "replay a complete, independently verified checkpoint without provenance; never add requests")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if !(*price > 0) {
		return fmt.Errorf("--input-price-per-million must be a positive account-specific rate")
	}
	records, err := benchmark.ReadDatasetRecords(*datasetPath)
	if err != nil {
		return err
	}
	criteria, err := readCriteria(*criteriaPath)
	if err != nil {
		return err
	}
	criteria, err = jev.EnrichCriteriaWithTrainingExamples(records, criteria, *examplesPerLabel, *exampleSeed)
	if err != nil {
		return err
	}
	criteriaHash, err := jev.CriteriaHash(criteria)
	if err != nil {
		return err
	}

	var evaluator jev.Evaluator
	if !*dryRun && os.Getenv("TYPESAFE_API_KEY") != "" {
		client, err := jev.NewClient(jev.ClientConfig{APIKey: os.Getenv("TYPESAFE_API_KEY")})
		if err != nil {
			return err
		}
		evaluator = client
	}
	report, err := jev.RunChoice(context.Background(), jev.ChoiceRunConfig{
		Evaluator:                   evaluator,
		Records:                     records,
		Criteria:                    criteria,
		Model:                       *model,
		Split:                       *split,
		Limit:                       *limit,
		SampleSeed:                  *sampleSeed,
		CriteriaVariant:             fmt.Sprintf("train-examples-%d", *examplesPerLabel),
		CriteriaHash:                criteriaHash,
		MaxCostUSD:                  *maxCost,
		InputPricePerMillion:        *price,
		MaxInputTokensPerRequest:    *maxTokens,
		CheckpointPath:              *checkpointPath,
		OutputPath:                  *outputPath,
		DryRun:                      *dryRun,
		AllowLegacyCheckpointReplay: *allowLegacy,
	})
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(report)
}

func runJevNoulRun(arguments []string) error {
	flags := flag.NewFlagSet("jev-noul-run", flag.ContinueOnError)
	datasetPath := flags.String("dataset", "data/goemotions/records.parquet", "input multi-label dataset Parquet path")
	criteriaPath := flags.String(
		"criteria",
		"configs/datasets/goemotions-noul-criteria.json",
		"path to a JSON object mapping emotions to Noul criteria",
	)
	outputPath := flags.String("output", "runs/jev-goemotions/validation.predictions.parquet", "prediction Parquet path")
	checkpointPath := flags.String("checkpoint", "runs/jev-goemotions/validation.checkpoint.jsonl", "durable JSONL checkpoint path")
	model := flags.String("model", "jev-1.13.0", "pinned TypeSafe model identifier")
	split := flags.String("split", benchmark.SplitValidation, "dataset split to classify")
	limit := flags.Int("limit", 56, "maximum records to include in this resumable run")
	sampleSeed := flags.String("sample-seed", "jev-goemotions-validation-probe-v1", "deterministic seed for a coverage-first limited sample")
	threshold := flags.Float64("threshold", 0.87, "validation-selected emotion probability threshold; neutral is emitted when none qualify")
	stateField := flags.String("state-field", "comment", "state field containing the record text")
	questionTemplate := flags.String(
		"question-template",
		"Does `comment` express %s?",
		"Noul question template with one %s label placeholder",
	)
	emptyPredictionPolicy := flags.String(
		"empty-prediction-policy",
		jev.NoulEmptyPredictionFallbackNeutral,
		"fallback-neutral or highest-probability",
	)
	maxStateCharacters := flags.Int(
		"max-state-characters",
		0,
		"maximum Unicode characters sent in the state; zero preserves full text",
	)
	stateTruncation := flags.String(
		"state-truncation",
		jev.NoulStateTruncationNone,
		"none or head-tail when a state character limit is set",
	)
	maxCost := flags.Float64("max-cost-usd", 0.50, "local hard ceiling in US dollars")
	price := flags.Float64("input-price-per-million", 0, "your account's price per million input tokens (required)")
	maxTokens := flags.Int("max-input-tokens-per-request", 4096, "local per-request token reservation and validation limit")
	dryRun := flags.Bool("dry-run", false, "show the safe request plan without calling Jev")
	allowLegacy := flags.Bool("allow-legacy-checkpoint-replay", false, "replay a complete, independently verified checkpoint without provenance; never add requests")
	if err := flags.Parse(arguments); err != nil {
		return err
	}
	if !(*price > 0) {
		return fmt.Errorf("--input-price-per-million must be a positive account-specific rate")
	}
	records, err := benchmark.ReadMultiLabelDatasetRecords(*datasetPath)
	if err != nil {
		return err
	}
	criteria, err := readNoulCriteria(*criteriaPath)
	if err != nil {
		return err
	}
	criteriaHash, err := jev.NoulCriteriaHash(criteria)
	if err != nil {
		return err
	}

	var evaluator jev.Evaluator
	if !*dryRun && os.Getenv("TYPESAFE_API_KEY") != "" {
		client, err := jev.NewClient(jev.ClientConfig{APIKey: os.Getenv("TYPESAFE_API_KEY")})
		if err != nil {
			return err
		}
		evaluator = client
	}
	report, err := jev.RunNoul(context.Background(), jev.NoulRunConfig{
		Evaluator:                   evaluator,
		Records:                     records,
		Criteria:                    criteria,
		Model:                       *model,
		Split:                       *split,
		Limit:                       *limit,
		SampleSeed:                  *sampleSeed,
		CriteriaHash:                criteriaHash,
		StateField:                  *stateField,
		QuestionTemplate:            *questionTemplate,
		EmptyPredictionPolicy:       *emptyPredictionPolicy,
		MaxStateCharacters:          *maxStateCharacters,
		StateTruncation:             *stateTruncation,
		Threshold:                   *threshold,
		MaxCostUSD:                  *maxCost,
		InputPricePerMillion:        *price,
		MaxInputTokensPerRequest:    *maxTokens,
		CheckpointPath:              *checkpointPath,
		OutputPath:                  *outputPath,
		DryRun:                      *dryRun,
		AllowLegacyCheckpointReplay: *allowLegacy,
	})
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(report)
}

func readCriteria(path string) (map[string]string, error) {
	criteriaBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read criteria: %w", err)
	}
	var criteria map[string]string
	if err := json.Unmarshal(criteriaBytes, &criteria); err != nil {
		return nil, fmt.Errorf("decode criteria JSON: %w", err)
	}
	if len(criteria) == 0 {
		return nil, fmt.Errorf("criteria JSON must contain at least one label")
	}
	for label, criterion := range criteria {
		if label == "" || criterion == "" {
			return nil, fmt.Errorf("criteria JSON contains an empty label or description")
		}
	}
	return criteria, nil
}

func readNoulCriteria(path string) (map[string]jev.NoulCriteria, error) {
	criteriaBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read Noul criteria: %w", err)
	}
	var criteria map[string]jev.NoulCriteria
	if err := json.Unmarshal(criteriaBytes, &criteria); err != nil {
		return nil, fmt.Errorf("decode Noul criteria JSON: %w", err)
	}
	if len(criteria) == 0 {
		return nil, fmt.Errorf("Noul criteria JSON must contain at least one label")
	}
	return criteria, nil
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
	  benchmark prepare-banking77 [--manifest configs/datasets/banking77.json] [--output data/banking77/records.parquet]
	  benchmark prepare-goemotions [--manifest configs/datasets/goemotions.json] [--output data/goemotions/records.parquet]
	  benchmark prepare-multieurlex [--manifest configs/datasets/multieurlex-english-level1.json] [--output data/multieurlex/records.parquet]
	  benchmark split --input records.parquet --output records-with-validation.parquet --seed split-2026
  benchmark metrics --dataset records.parquet --predictions predictions.parquet [--split test]
	  benchmark metrics-multilabel --dataset records.parquet --predictions predictions.parquet [--split test]
	  benchmark jev-choice --text "..." [--criteria labels.json] [--model jev-latest]
	  benchmark jev-run --input-price-per-million RATE [--dry-run] [--limit 25] [--max-cost-usd 0.50]
	  benchmark jev-noul-run --input-price-per-million RATE [--dry-run] [--limit 56] [--threshold 0.87] [--max-cost-usd 0.50]`)
}
