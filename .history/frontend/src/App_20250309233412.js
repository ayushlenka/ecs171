import React, { useState } from "react";

function App() {
    const [tweet, setTweet] = useState("");
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const analyzeTweet = async () => {
        if (!tweet.trim()) {
            setError("Please enter a tweet.");
            return;
        }
        setError("");
        setLoading(true);

        try {
            const response = await fetch("http://127.0.0.1:8000/analyze_tweet", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ tweet })
            });

            if (!response.ok) {
                throw new Error("Failed to fetch response");
            }

            const data = await response.json();
            setResult(data);
        } catch (error) {
            setError("Error analyzing tweet. Try again.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
            <h2>Stock Sentiment Analysis</h2>
            <input
                type="text"
                placeholder="Enter a tweet about a stock..."
                value={tweet}
                onChange={(e) => setTweet(e.target.value)}
                style={{ width: "60%", padding: "10px", marginBottom: "10px" }}
            />
            <br />
            <button onClick={analyzeTweet} disabled={loading} style={{ padding: "10px 20px", cursor: "pointer" }}>
                {loading ? "Analyzing..." : "Analyze Tweet"}
            </button>

            {error && <p style={{ color: "red" }}>{error}</p>}

            {result && (
                <div style={{ marginTop: "20px" }}>
                    <h3>Results:</h3>
                    <p><strong>Emotion:</strong> {result.emotion}</p>
                    <p><strong>Market Sentiment:</strong> {result.sentiment}</p>
                </div>
            )}
        </div>
    );
}

export default App;
