import React, { useState } from "react";

function App() {
  const [userInput, setUserInput] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeTweet = async () => {
    setLoading(true);
    setError("");
    setAnalysis(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/analyze_tweet", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ tweet: userInput }),
      });

      if (!response.ok) {
        throw new Error(`HTTP Error! Status: ${response.status}`);
      }

      const data = await response.json();
      setAnalysis(data);
    } catch (error) {
      console.error("Error fetching analysis:", error);
      setError("Failed to analyze tweet. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-5">
      <h1 className="text-2xl font-bold mb-4">Tweet Sentiment Analyzer</h1>

      <textarea
        className="w-96 p-3 border rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        rows="3"
        placeholder="Enter a tweet..."
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
      />

      <button
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-md shadow-md hover:bg-blue-600 transition duration-300"
        onClick={analyzeTweet}
        disabled={loading || !userInput.trim()}
      >
        {loading ? "Analyzing..." : "Analyze Tweet"}
      </button>

      {error && <p className="mt-3 text-red-500">{error}</p>}

      {analysis && (
        <div className="mt-6 p-4 border rounded-md bg-white shadow-md w-96">
          <h2 className="text-lg font-semibold">Analysis Result</h2>
          <p><strong>Emotion:</strong> {analysis.emotion}</p>
          <p><strong>Sentiment:</strong> {analysis.sentiment}</p>
        </div>
      )}
    </div>
  );
}

export default App;
