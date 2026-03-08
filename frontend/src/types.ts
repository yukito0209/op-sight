export type Sentiment = "negative" | "neutral" | "positive";

export interface SentimentResult {
  sentiment: Sentiment;
  sentiment_zh: string;
  confidence: number;
  intensity: number;
  category: string;
  keywords: string[];
  is_sarcasm: boolean;
  sarcasm_reason: string | null;
  probabilities: Record<Sentiment, number>;
  suggested_action: string;
  processing_time_ms: number;
}

export interface BatchResult {
  results: SentimentResult[];
  total_count: number;
  avg_confidence: number;
  sentiment_distribution: Record<Sentiment, number>;
}
