import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

export const analyzeTweet = async (tweet) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/analyze_tweet`, {
            tweet,
        }, {
            headers: { "Content-Type": "application/json" } 
        });

        return response.data;
    } catch (error) {
        console.error("API Error:", error.response ? error.response.data : error.message);
        return null;
    }
};
