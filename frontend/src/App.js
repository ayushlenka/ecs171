import React, { useState } from "react";
import { analyzeTweet } from "./services/api";

function App() {
    const [tweet, setTweet] = useState("");
    const [result, setResult] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const response = await analyzeTweet(tweet);
        setResult(response);
    };

    return (
        <div style={{ textAlign: "center", padding: "20px" }}>
            <h1>Stock Tweet Sentiment Analyzer</h1>
            <form onSubmit={handleSubmit}>
                <input 
                    type="text" 
                    value={tweet} 
                    onChange={(e) => setTweet(e.target.value)} 
                    placeholder="Enter a tweet"
                    required
                    style={{ width: "60%", padding: "10px" }}
                />
                <button type="submit" style={{ marginLeft: "10px", padding: "10px 20px" }}>
                    Analyze
                </button>
            </form>

            {result && (
                <div style={{ marginTop: "20px" }}>
                    <h3>Result:</h3>
                    <p><strong>Emotion:</strong> {result.emotion}</p>
                    <p><strong>Sentiment:</strong> {result.sentiment}</p>
                </div>
            )}
        </div>
    );
}

export default App;
