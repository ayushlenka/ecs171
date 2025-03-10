import React, { useState } from "react";

function App() {
    const [tweetInput, setTweetInput] = useState("");
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleAnalyze = async () => {
        setError(null); // Clear previous errors
        setLoading(true);

        try {
            const response = await fetch("http://127.0.0.1:8000/analyze_tweet", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ tweet: tweetInput }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Unknown error occurred");
            }

            setResult(data); // Set the result to display
        } catch (error) {
            console.error("Error analyzing tweet:", error);
            setError(error.message); // Show error message
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ textAlign: "center", padding: "50px", fontFamily: "Arial, sans-serif" }}>
            <h1>Stock Tweet Sentiment Analyzer</h1>
            <textarea
                placeholder="Enter a tweet about a stock..."
                value={tweetInput}
                onChange={(e) => setTweetInput(e.target.value)}
                rows="3"
                cols="50"
                style={{ padding: "10px", fontSize: "16px", marginBottom: "10px" }}
            />
            <br />
            <button
                onClick={handleAnalyze}
                disabled={loading}
                style={{
                    padding: "10px 20px",
                    fontSize: "16px",
                    cursor: "pointer",
                    backgroundColor: "#007BFF",
                    color: "#fff",
                    border: "none",
                    borderRadius: "5px",
                }}
            >
                {loading ? "Analyzing..." : "Analyze Tweet"}
            </button>
            <br /><br />
            {error && <p style={{ color: "red" }}>Error: {error}</p>}
            {result && (
                <div style={{ marginTop: "20px" }}>
                    <h2>Analysis Result:</h2>
                    <p><strong>Emotion:</strong> {result.emotion}</p>
                    <p><strong>Sentiment:</strong> {result.sentiment}</p>
                </div>
            )}
        </div>
    );
}

export default App;
