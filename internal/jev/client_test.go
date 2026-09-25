package jev

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestEvaluateSendsTypedRequest(t *testing.T) {
	var request SystemOneRequest
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, rawRequest *http.Request) {
		if rawRequest.URL.Path != "/v1/systemone" {
			t.Fatalf("path = %q", rawRequest.URL.Path)
		}
		if authorization := rawRequest.Header.Get("Authorization"); authorization != "Bearer test-key" {
			t.Fatalf("authorization = %q", authorization)
		}
		if err := json.NewDecoder(rawRequest.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = writer.Write([]byte(`{
			"model":"jev-1.13.0",
			"answers":{"intent":{"type":"choice","choice":"billing","probabilities":{"billing":0.9,"technical":0.1},"confidence":0.8}},
			"usage":{"input_tokens":12,"output_tokens":3}
		}`))
	}))
	defer server.Close()

	client, err := NewClient(ClientConfig{APIKey: "test-key", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	response, err := client.Evaluate(context.Background(), SystemOneRequest{
		State: "My card payment was declined.",
		Model: "jev-latest",
		Questions: map[string]Question{
			"intent": {
				Type:         "choice",
				Instructions: "Choose the banking intent.",
				Criteria:     map[string]string{"billing": "Payment question", "technical": "App defect"},
			},
		},
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if request.Model != "jev-latest" || response.Answers["intent"].Choice != "billing" {
		t.Fatalf("unexpected request or response: request=%+v response=%+v", request, response)
	}
}

func TestEvaluateRetriesRateLimit(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		attempts++
		if attempts == 1 {
			writer.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = writer.Write([]byte(`{"model":"jev-1.13.0","answers":{},"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	client, err := NewClient(ClientConfig{
		APIKey: "test-key", BaseURL: server.URL, MaxAttempts: 2, InitialBackoff: time.Nanosecond,
	})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	_, err = client.Evaluate(context.Background(), validRequest())
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2", attempts)
	}
}

func TestEvaluateDoesNotRetryBadRequest(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		attempts++
		http.Error(writer, "invalid request", http.StatusUnprocessableEntity)
	}))
	defer server.Close()

	client, err := NewClient(ClientConfig{APIKey: "test-key", BaseURL: server.URL})
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}
	_, err = client.Evaluate(context.Background(), validRequest())
	if err == nil {
		t.Fatal("Evaluate() returned nil error")
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1", attempts)
	}
}

func validRequest() SystemOneRequest {
	return SystemOneRequest{
		State: "Example text",
		Model: "jev-latest",
		Questions: map[string]Question{
			"label": {Type: "noul", Instructions: "Does this match?"},
		},
	}
}
