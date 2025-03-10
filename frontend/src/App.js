import React, { useState } from "react";

function App() {
  const [userInput, setUserInput] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeTweet = async () => {
    setLoading(true);
    setError("");
    setAnalysis(null);
    setPrediction(null);

    try {
      console.log("🔄 Sending tweet for sentiment analysis...");

      const response = await fetch("http://127.0.0.1:8000/analyze_tweet", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tweet: userInput }),
      });

      if (!response.ok) {
        throw new Error(`HTTP Error! Status: ${response.status}`);
      }

      const data = await response.json();
      console.log("✅ Sentiment analysis result:", data);
      setAnalysis(data);

      await predictStock(data.sentiment, userInput);
    } catch (error) {
      setError("Failed to analyze tweet. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const predictStock = async (sentiment, tweet) => {
    try {
      console.log("🔄 Sending sentiment score to stock predictor...");

      let sentiment_score = sentiment.toLowerCase() === "bullish" ? 1.0 : sentiment.toLowerCase() === "bearish" ? -1.0 : 0.0;
      console.log("📤 Sentiment Score Sent:", sentiment_score);

      let company = extractCompany(tweet);
      console.log("📤 Sending Data:", { company, sentiment_score });

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ company, sentiment_score }),
      });

      if (!response.ok) {
        throw new Error(`HTTP Error! Status: ${response.status}`);
      }

      const data = await response.json();

      setPrediction({
        company: data.company,
        predicted_change: data.predicted_change,
        trend: data.trend,
      });
    } catch (error) {
      setError("Failed to predict stock. Please try again.");
    }
  };

  const extractCompany = (tweet) => {
    const companies = ["Tesla", "Apple", "Amazon", "Microsoft"];
    return companies.find((company) => tweet.toLowerCase().includes(company.toLowerCase())) || "Tesla";
  };

  // ✅ Embedded styles for cleaner UI
  const styles = {
    appContainer: {
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "100vh",
      padding: "20px",
      backgroundColor: "#f4f4f4",
      fontFamily: "Arial, sans-serif",
    },
    title: {
      fontSize: "24px",
      fontWeight: "bold",
      color: "#333",
      marginBottom: "20px",
    },
    inputBox: {
      width: "80%",
      maxWidth: "400px",
      padding: "10px",
      fontSize: "16px",
      border: "1px solid #ccc",
      borderRadius: "5px",
      resize: "none",
    },
    analyzeBtn: {
      marginTop: "10px",
      padding: "10px 20px",
      fontSize: "16px",
      backgroundColor: "#007bff",
      color: "white",
      border: "none",
      borderRadius: "5px",
      cursor: "pointer",
      transition: "background 0.3s ease",
    },
    analyzeBtnHover: {
      backgroundColor: "#0056b3",
    },
    resultBox: {
      marginTop: "20px",
      padding: "15px",
      width: "80%",
      maxWidth: "400px",
      backgroundColor: "white",
      borderRadius: "8px",
      boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)",
    },
    errorMessage: {
      color: "red",
      marginTop: "10px",
    },
    trend: {
      fontWeight: "bold",
    },
    positive: { color: "green" },
    negative: { color: "red" },
    stable: { color: "gray" },
  };

  return (
    <div style={styles.appContainer}>
      <h1 style={styles.title}>Tweet Sentiment & Stock Predictor</h1>

      <textarea
        style={styles.inputBox}
        rows="3"
        placeholder="Enter a tweet..."
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
      />

      <button
        style={styles.analyzeBtn}
        onClick={analyzeTweet}
        disabled={loading || !userInput.trim()}
      >
        {loading ? "Analyzing..." : "Analyze & Predict"}
      </button>

      {error && <p style={styles.errorMessage}>{error}</p>}

      {analysis && (
        <div style={styles.resultBox}>
          <h2>Sentiment Analysis</h2>
          <p><strong>Emotion:</strong> {analysis.emotion}</p>
          <p><strong>Sentiment:</strong> {analysis.sentiment}</p>
        </div>
      )}

      {prediction && (
        <div style={styles.resultBox}>
          <h2>Stock Prediction</h2>
          <p><strong>Company:</strong> {prediction.company}</p>
          <p><strong>Predicted Change:</strong> { (prediction.predicted_change * 100).toFixed(2) }%</p>
          <p>
            <strong>Trend:</strong>
            <span style={ prediction.trend === "Positive" ? styles.positive :
                          prediction.trend === "Negative" ? styles.negative :
                          styles.stable }>
              {" "}{prediction.trend}
            </span>
          </p>
        </div>
      )}
    </div>
  );
}

export default App;
