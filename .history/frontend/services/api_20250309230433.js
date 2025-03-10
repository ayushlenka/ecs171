import axios from "axios";

const API_BASE_URL = "http://127.0.0.1:8000"; // FastAPI URL

export const analyzeTweet = async (tweet) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/analyze_tweet`, { tweet });
        return response.data;
    } catch (error) {
        console.error("API Error:", error);
        return null;
    }
};
