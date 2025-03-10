import React, { useState } from "react";
import axios from "axios";

function App() {
    const [tweet, setTweet] = useState("");
    const [result, setResult] = useState(null);

    const analyzeTweet = async () => {
        try {
            const response = await axios.post(
                "http://127.0.0.1:8000/analyze_tweet",
                { tweet },
                {
                    headers: {
                        "Content-Type": "application/json"
                    }
                }
            );
            setResult(response.data);
        } catch (error) {
            console.error("Error:", error);
        }
    };

    return (
        <div>
            <h1>Stock Sentiment Analyzer</h1>
            <textarea
                value={tweet}
                onChange={(e) => setTweet(e.target.value)}
                placeholder="Enter a tweet..."
            />
            <button onClick={analyzeTweet}>Analyze</button>

            {result && (
                <div>
                    <h3>Results:</h3>
                    <p>Emotion: {result.emotion}</p>
                    <p>Sentiment: {result.sentiment}</p>
                </div>
            )}
        </div>
    );
}

export default App;
