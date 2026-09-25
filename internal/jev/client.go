package jev

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const defaultBaseURL = "https://api.typesafe.ai"

type ClientConfig struct {
	APIKey         string
	BaseURL        string
	HTTPClient     *http.Client
	MaxAttempts    int
	InitialBackoff time.Duration
}

type Client struct {
	apiKey         string
	baseURL        string
	httpClient     *http.Client
	maxAttempts    int
	initialBackoff time.Duration
}

type SystemOneRequest struct {
	State     any                 `json:"state"`
	Model     string              `json:"model"`
	Questions map[string]Question `json:"questions"`
}

type Question struct {
	Type         string `json:"type"`
	Instructions any    `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

type SystemOneResponse struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   Usage             `json:"usage"`
}

type Answer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice,omitempty"`
	Noul          *float64           `json:"noul,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type APIError struct {
	StatusCode int
	Body       string
}

func (error APIError) Error() string {
	return fmt.Sprintf("typesafe API returned HTTP %d: %s", error.StatusCode, error.Body)
}

func NewClient(config ClientConfig) (*Client, error) {
	if strings.TrimSpace(config.APIKey) == "" {
		return nil, errors.New("TypeSafe API key must not be empty")
	}
	if config.BaseURL == "" {
		config.BaseURL = defaultBaseURL
	}
	if config.HTTPClient == nil {
		config.HTTPClient = &http.Client{Timeout: 30 * time.Second}
	}
	if config.MaxAttempts == 0 {
		config.MaxAttempts = 3
	}
	if config.MaxAttempts < 1 {
		return nil, fmt.Errorf("max attempts must be at least 1, got %d", config.MaxAttempts)
	}
	if config.InitialBackoff == 0 {
		config.InitialBackoff = time.Second
	}
	if config.InitialBackoff < 0 {
		return nil, fmt.Errorf("initial backoff must not be negative")
	}
	return &Client{
		apiKey:         config.APIKey,
		baseURL:        strings.TrimRight(config.BaseURL, "/"),
		httpClient:     config.HTTPClient,
		maxAttempts:    config.MaxAttempts,
		initialBackoff: config.InitialBackoff,
	}, nil
}

func (client *Client) Evaluate(ctx context.Context, request SystemOneRequest) (SystemOneResponse, error) {
	if err := validateRequest(request); err != nil {
		return SystemOneResponse{}, err
	}
	body, err := json.Marshal(request)
	if err != nil {
		return SystemOneResponse{}, fmt.Errorf("encode TypeSafe request: %w", err)
	}

	for attempt := 0; attempt < client.maxAttempts; attempt++ {
		response, retryAfter, err := client.evaluateOnce(ctx, body)
		if err == nil {
			return response, nil
		}
		var apiError APIError
		if !errors.As(err, &apiError) || (apiError.StatusCode != http.StatusTooManyRequests && apiError.StatusCode != 529) {
			return SystemOneResponse{}, err
		}
		if attempt == client.maxAttempts-1 {
			return SystemOneResponse{}, err
		}

		delay := retryAfter
		if delay == 0 {
			delay = client.initialBackoff * time.Duration(1<<attempt)
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return SystemOneResponse{}, ctx.Err()
		case <-timer.C:
		}
	}
	return SystemOneResponse{}, errors.New("unreachable retry state")
}

func (client *Client) evaluateOnce(ctx context.Context, body []byte) (SystemOneResponse, time.Duration, error) {
	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		client.baseURL+"/v1/systemone",
		bytes.NewReader(body),
	)
	if err != nil {
		return SystemOneResponse{}, 0, fmt.Errorf("create TypeSafe request: %w", err)
	}
	request.Header.Set("Authorization", "Bearer "+client.apiKey)
	request.Header.Set("Content-Type", "application/json")

	response, err := client.httpClient.Do(request)
	if err != nil {
		return SystemOneResponse{}, 0, fmt.Errorf("send TypeSafe request: %w", err)
	}
	defer response.Body.Close()

	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		body, readErr := io.ReadAll(io.LimitReader(response.Body, 4096))
		if readErr != nil {
			return SystemOneResponse{}, 0, fmt.Errorf("read TypeSafe error response: %w", readErr)
		}
		return SystemOneResponse{}, parseRetryAfter(response.Header.Get("Retry-After")), APIError{
			StatusCode: response.StatusCode,
			Body:       strings.TrimSpace(string(body)),
		}
	}

	var decoded SystemOneResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		return SystemOneResponse{}, 0, fmt.Errorf("decode TypeSafe response: %w", err)
	}
	if decoded.Model == "" {
		return SystemOneResponse{}, 0, errors.New("TypeSafe response did not include a model")
	}
	return decoded, 0, nil
}

func validateRequest(request SystemOneRequest) error {
	if strings.TrimSpace(request.Model) == "" {
		return errors.New("TypeSafe model must not be empty")
	}
	if len(request.Questions) == 0 {
		return errors.New("at least one TypeSafe question is required")
	}
	for id, question := range request.Questions {
		if strings.TrimSpace(id) == "" {
			return errors.New("TypeSafe question id must not be empty")
		}
		if question.Type != "choice" && question.Type != "noul" && question.Type != "score" {
			return fmt.Errorf("question %q has unsupported type %q", id, question.Type)
		}
		if question.Instructions == nil {
			return fmt.Errorf("question %q must include instructions", id)
		}
	}
	return nil
}

func parseRetryAfter(value string) time.Duration {
	seconds, err := strconv.Atoi(value)
	if err != nil || seconds < 0 {
		return 0
	}
	return time.Duration(seconds) * time.Second
}
