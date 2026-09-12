package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

type predictRequest struct {
	Team1        string `json:"team1"`
	Team2        string `json:"team2"`
	Venue        string `json:"venue"`
	TossWinner   string `json:"toss_winner"`
	TossDecision string `json:"toss_decision"`
	MatchDate    string `json:"match_date"`
}

type predictResponse struct {
	Winner           string  `json:"winner"`
	Team1Probability float64 `json:"team1_probability"`
	Team2Probability float64 `json:"team2_probability"`
	Message          string  `json:"message"`
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/predict", predictHandler)
	mux.HandleFunc("/", rootHandler)

	server := &http.Server{
		Addr:         ":8080",
		Handler:      withCORS(mux),
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	log.Println("Backend running on http://localhost:8080")
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func rootHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = fmt.Fprint(w, "Match Winner Prediction API")
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Only POST is allowed", http.StatusMethodNotAllowed)
		return
	}

	var req predictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}

	req.Team1 = strings.TrimSpace(req.Team1)
	req.Team2 = strings.TrimSpace(req.Team2)
	req.Venue = strings.TrimSpace(req.Venue)
	req.TossWinner = strings.TrimSpace(req.TossWinner)
	req.TossDecision = strings.TrimSpace(req.TossDecision)

	if req.Team1 == "" || req.Team2 == "" || req.Venue == "" || req.TossWinner == "" || req.TossDecision == "" {
		http.Error(w, "team1, team2, venue, toss_winner and toss_decision are required", http.StatusBadRequest)
		return
	}

	if req.Team1 == req.Team2 {
		http.Error(w, "team1 and team2 must be different", http.StatusBadRequest)
		return
	}

	if req.MatchDate == "" {
		req.MatchDate = time.Now().Format(time.RFC3339)
	}

	payload, err := json.Marshal(req)
	if err != nil {
		http.Error(w, "Could not prepare prediction request", http.StatusInternalServerError)
		return
	}

	pythonBin, err := locatePython()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	modelScript := filepath.Join(projectRoot(), "ml_pipeline", "model", "predict_model.py")
	cmd := exec.Command(pythonBin, modelScript, string(payload))
	cmd.Dir = projectRoot()

	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Prediction script error: %v\n%s", err, string(output))
		http.Error(w, "Model prediction failed", http.StatusInternalServerError)
		return
	}

	var resp predictResponse
	if err := json.Unmarshal(output, &resp); err != nil {
		log.Printf("Response parse error: %v\nOutput: %s", err, string(output))
		http.Error(w, "Could not parse model output", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func projectRoot() string {
	_, filename, _, _ := runtime.Caller(0)
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "..", ".."))
}

func locatePython() (string, error) {
	candidates := []string{
		filepath.Join(projectRoot(), ".venv", "bin", "python"),
		"/usr/bin/python3",
		"/usr/local/bin/python3",
		"python3",
	}

	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("Python executable not found in the project environment")
}
