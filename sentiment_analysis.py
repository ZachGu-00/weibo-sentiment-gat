import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc
import warnings
warnings.filterwarnings('ignore')


class ChineseWeiboSentimentAnalyzer:
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tokenizers = {}
        
        # Model configurations with weights
        self.model_configs = [
            {
                'name': 'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment',
                'weight': 0.5,
                'type': 'sentiment'
            },
            {
                'name': 'techthiyanes/chinese_sentiment',
                'weight': 0.3,
                'type': 'sentiment'
            },
            {
                'name': 'uer/roberta-base-finetuned-chinanews-chinese',
                'weight': 0.2,
                'type': 'news'
            }
        ]
    
    def load_models(self):
        successful_models = 0
        
        for config in self.model_configs:
            model_name = config['name']
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model.to(self.device)
                model.eval()
                
                self.tokenizers[model_name] = tokenizer
                self.models[model_name] = model
                successful_models += 1
            except Exception as e:
                continue
        
        if successful_models == 0:
            raise Exception("No sentiment analysis models loaded successfully!")
    
    def calculate_sentiment_batch(self, texts, batch_size=256):
        if not texts or len(texts) == 0:
            return []
        
        all_scores = []
        
        # Calculate sentiment scores for each model
        for config in self.model_configs:
            model_name = config['name']
            weight = config['weight']
            
            if model_name not in self.models:
                continue
            
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            model_scores = []
            
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Handle empty texts
                batch_texts = [text if text and str(text).strip() else "。" for text in batch_texts]
                
                try:
                    # Encode texts
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=128  # Weibo posts are usually short
                    ).to(self.device)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                        
                        # Handle different model output classes
                        if logits.shape[1] == 2:
                            # Binary classification: negative/positive
                            probs = torch.softmax(logits, dim=-1)
                            scores = probs[:, 1].cpu().numpy()  # Positive probability
                        elif logits.shape[1] == 3:
                            # Three classes: negative/neutral/positive
                            probs = torch.softmax(logits, dim=-1)
                            # Calculate composite sentiment score: negative*0 + neutral*0.5 + positive*1
                            scores = (probs[:, 0] * 0 + probs[:, 1] * 0.5 + probs[:, 2] * 1).cpu().numpy()
                        else:
                            # Other cases, use max probability class
                            probs = torch.softmax(logits, dim=-1)
                            scores = probs.max(dim=-1)[0].cpu().numpy()
                    
                    model_scores.extend(scores.tolist())
                
                except Exception as e:
                    # Return neutral scores on error
                    model_scores.extend([0.5] * len(batch_texts))
                
                # Clean GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            all_scores.append((model_scores, weight))
        
        # Weighted average of all model results
        final_scores = np.zeros(len(texts))
        total_weight = 0
        
        for scores, weight in all_scores:
            final_scores += np.array(scores) * weight
            total_weight += weight
        
        if total_weight > 0:
            final_scores /= total_weight
        
        return final_scores.tolist()


class SentimentAnalyzer:
    
    def __init__(self, network_path):
        self.network_path = network_path
        self.G = None
        self.weibo_analyzer = None
    
    def load_network(self):
        with open(self.network_path, 'rb') as f:
            self.G = pickle.load(f)
    
    def load_sentiment_model(self):
        self.weibo_analyzer = ChineseWeiboSentimentAnalyzer()
        self.weibo_analyzer.load_models()
    
    def add_sentiment_features(self):
        self.load_sentiment_model()
        
        node_count = 0
        successful_analyses = 0
        no_text_count = 0
        all_scores = []
        
        for user_name, node_data in tqdm(self.G.nodes(data=True), desc="Computing sentiment"):
            node_count += 1
            
            try:
                texts = node_data.get('texts', [])
                
                if texts and len(texts) > 0:
                    sentiment_scores = self.weibo_analyzer.calculate_sentiment_batch(texts, batch_size=32)
                    avg_sentiment = float(np.mean(sentiment_scores))
                    
                    if len(sentiment_scores) > 1:
                        sentiment_volatility = float(np.std(sentiment_scores))
                    else:
                        sentiment_volatility = 0.0
                    
                    all_scores.append(avg_sentiment)
                    successful_analyses += 1
                    
                    self.G.nodes[user_name]['sentiment_score'] = avg_sentiment
                    self.G.nodes[user_name]['sentiment_volatility'] = sentiment_volatility
                    
                    if avg_sentiment < 0.3:
                        sentiment_label = 'negative'
                    elif avg_sentiment > 0.7:
                        sentiment_label = 'positive'
                    else:
                        sentiment_label = 'neutral'
                    
                    self.G.nodes[user_name]['sentiment_label'] = sentiment_label
                
                else:
                    self.G.nodes[user_name]['sentiment_score'] = 0.5
                    self.G.nodes[user_name]['sentiment_volatility'] = 0.0
                    self.G.nodes[user_name]['sentiment_label'] = 'neutral'
                    no_text_count += 1
            
            except Exception as e:
                self.G.nodes[user_name]['sentiment_score'] = 0.5
                self.G.nodes[user_name]['sentiment_volatility'] = 0.0
                self.G.nodes[user_name]['sentiment_label'] = 'neutral'
            
            if node_count % 1000 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def analyze_sentiment_distribution(self):
        """Analyze sentiment distribution."""
        sentiment_data = []
        for user_name, node_data in self.G.nodes(data=True):
            sentiment_data.append({
                'user_name': user_name,
                'sentiment_score': node_data.get('sentiment_score', 0.5),
                'sentiment_volatility': node_data.get('sentiment_volatility', 0.0),
                'sentiment_label': node_data.get('sentiment_label', 'neutral'),
                'text_count': node_data.get('text_count', 0),
                'post_count': node_data.get('post_count', 0),
                'comment_count': node_data.get('comment_count', 0),
                'total_posts': node_data.get('total_posts', 0),
                'activity_days': node_data.get('activity_days', 0)
            })
        
        return pd.DataFrame(sentiment_data)
    
    def save_updated_network(self, output_path=None):
        save_path = output_path or self.network_path
        with open(save_path, 'wb') as f:
            pickle.dump(self.G, f)
    
    def run_sentiment_analysis(self, output_path=None, results_csv_path=None):

        self.load_network()
        self.add_sentiment_features()
        sentiment_df = self.analyze_sentiment_distribution()
        self.save_updated_network(output_path)
        
        if results_csv_path:
            sentiment_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
        
        return self.G, sentiment_df


def main():
    network_path = "weibo_user_interaction_network.pkl"
    
    analyzer = SentimentAnalyzer(network_path)
    G, sentiment_df = analyzer.run_sentiment_analysis(
        results_csv_path="sentiment_analysis_results.csv"
    )
    
    return G, sentiment_df


if __name__ == "__main__":
    G, sentiment_df = main()