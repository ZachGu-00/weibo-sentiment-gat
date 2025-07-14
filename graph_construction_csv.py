import os
import pandas as pd
import numpy as np
import time
import requests
import re
import json
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import jieba
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import networkx as nx
import pickle
from tqdm import tqdm
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Config:
    """配置类"""
    api_key: str = "sk-8a5efb9f369b4f1d846110de379a3753"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus-latest"
    input_folder: str = r"E:\ABM-LLM-Simulation-project\comments_split"
    output_folder: str = r"E:\ABM-LLM-Simulation-project\subgraph"
    batch_folder: str = r"E:\ABM-LLM-Simulation-project\batch_temp"  # 批量处理临时文件夹
    start_index: int = 450 
    end_index: int = 1000   
    api_timeout: int = 30
    api_delay: float = 0.2
    semantic_model_name: str = "shibing624/text2vec-base-chinese"
    log_level: str = "INFO"
    max_workers: int = 3
    use_batch_api: bool = True  # 是否使用批量API
    batch_size: int = 5000  # 每个批次的最大请求数
    # 图构建相关配置
    video_publish_date: datetime = datetime(2025, 6, 19)
    data_collection_date: datetime = datetime(2025, 7, 1)
    alpha: float = 0.1  # 点赞权重系数


class Logger:
    """日志管理类"""
    
    @staticmethod
    def setup_logger(name: str = __name__, level: str = "INFO") -> logging.Logger:
        """设置日志配置"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        if not logger.handlers:
            file_handler = logging.FileHandler('interaction_graph_analysis.log', encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger


class BatchAPIManager:
    """批量API管理器"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        # 创建批量处理临时文件夹
        Path(config.batch_folder).mkdir(parents=True, exist_ok=True)
    
    def create_batch_request(self, custom_id: str, parent_comment: str, reply_comment: str) -> dict:
        """创建单个批量请求"""
        prompt = f"""你是一个社交媒体评论分析专家。请分析"回复"相对于"父评论"的关系，并以JSON格式输出你的分析结果，包含'stance'和'style'两个维度。

    - 'stance'维度选项（单选）: ["支持", "反对", "中立/无关"]
    - 'style'维度选项（可多选）: ["逻辑论证", "举例说明", "提出质疑", "幽默/玩梗", "信息补充"]

    父评论: {parent_comment}
    回复: {reply_comment}

    请严格按照以下格式输出JSON:
    {{"stance": "选择一个", "style": ["可以选择多个"]}}"""
        
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",  # 修改这里：从 "/v1/chat/completions" 确保正确
            "body": {
                "model": self.config.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 100
            }
        }
    
    def write_batch_file(self, requests: List[dict], filepath: str) -> None:
        """将批量请求写入JSONL文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for request in requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
    
    def upload_file(self, filepath: str) -> str:
        """上传文件并返回文件ID"""
        with open(filepath, 'rb') as f:
            response = self.client.files.create(
                file=f,
                purpose="batch"
            )
        return response.id
    
    def create_batch_job(self, file_id: str) -> str:
        """创建批量任务并返回任务ID"""
        batch = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                'ds_name': f"评论立场分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'ds_description': '批量分析评论互动的立场和风格'
            }
        )
        return batch.id
    
    def wait_for_batch_completion(self, batch_id: str, check_interval: int = 30) -> dict:
        """等待批量任务完成"""
        start_time = time.time()
        check_count = 0
        
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            
            # 获取进度信息
            request_counts = batch.request_counts
            total_requests = request_counts.total
            completed_requests = request_counts.completed
            failed_requests = request_counts.failed
            
            # 计算进度百分比
            progress_percent = (completed_requests / total_requests * 100) if total_requests > 0 else 0
            
    
    def download_results(self, output_file_id: str, filepath: str) -> None:
        """下载结果文件"""
        content = self.client.files.content(output_file_id)
        with open(filepath, 'wb') as f:
            f.write(content.read())
    
    def parse_batch_results(self, filepath: str) -> Dict[str, Tuple[str, List[str]]]:
        """解析批量结果文件"""
        results = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    custom_id = data['custom_id']
                    
                    if data['response']['status_code'] == 200:
                        content = data['response']['body']['choices'][0]['message']['content']
                        stance, styles = self._parse_stance_response(content)
                        results[custom_id] = (stance, styles)
                    else:
                        self.logger.warning(f"请求失败 {custom_id}: {data.get('error', 'Unknown error')}")
                        results[custom_id] = ("中立/无关", ["信息补充"])
        
        return results
    
    def _parse_stance_response(self, content: str) -> Tuple[str, List[str]]:
        """解析立场分析响应"""
        try:
            cleaned_content = content.strip()
            
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            
            cleaned_content = cleaned_content.strip()
            
            analysis = json.loads(cleaned_content)
            stance = analysis.get("stance", "中立/无关")
            styles = analysis.get("style", ["信息补充"])
            
            # 验证立场类别
            valid_stances = ["支持", "反对", "中立/无关"]
            if stance not in valid_stances:
                stance = "中立/无关"
            
            # 验证互动方式
            valid_styles = ["逻辑论证", "举例说明", "提出质疑", "幽默/玩梗", "信息补充"]
            styles = [style for style in styles if style in valid_styles]
            if not styles:
                styles = ["信息补充"]
            
            return stance, styles
            
        except Exception as e:
            self.logger.warning(f"JSON解析失败: {str(e)}")
            return "中立/无关", ["信息补充"]


class SemanticAnalyzer:
    """语义分析器"""
    
    def __init__(self, model_name: str, logger: logging.Logger):
        self.logger = logger
        self.sentence_model = None
        self.tfidf_vectorizer = None
        self._initialize_models(model_name)
    
    def _initialize_models(self, model_name: str) -> None:
        """初始化语义模型"""
        try:
            self.logger.info("正在加载中文语义模型...")
            self.sentence_model = SentenceTransformer(model_name)
            self.logger.info("中文语义模型加载完成")
        except Exception as e:
            self.logger.warning(f"语义模型加载失败: {str(e)}, 将使用备用TF-IDF方法")
            self.sentence_model = None
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                tokenizer=self._tokenize_chinese,
                lowercase=False
            )
    
    @staticmethod
    def _tokenize_chinese(text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        try:
            if self.sentence_model is not None:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity_matrix = self.sentence_model.similarity(embeddings, embeddings)
                return float(similarity_matrix[0][1])
            else:
                texts = [text1, text2]
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(cosine_sim)
        except Exception as e:
            self.logger.warning(f"语义相似度计算失败: {str(e)}")
            return 0.0
    
    def encode_text(self, text: str) -> np.ndarray:
        """获取文本嵌入"""
        try:
            if self.sentence_model is not None:
                return self.sentence_model.encode(text)
            else:
                return self.tfidf_vectorizer.transform([text]).toarray()[0]
        except Exception as e:
            self.logger.warning(f"文本编码失败: {str(e)}")
            return np.zeros(768)


class LLMStanceAnalyzer:
    """LLM立场分析器 - 支持实时和批量模式"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        self.stance_prompt = self._build_stance_prompt()
    
    def _build_stance_prompt(self) -> str:
        """构建立场分析提示词"""
        return """你是一个社交媒体评论分析专家。请分析"回复"相对于"父评论"的关系，并以JSON格式输出你的分析结果，包含'stance'和'style'两个维度。

- 'stance'维度选项（单选）: ["支持", "反对", "中立/无关"]
- 'style'维度选项（可多选）: ["逻辑论证", "举例说明", "提出质疑", "幽默/玩梗", "信息补充"]

父评论: {parent_comment}
回复: {reply_comment}

请严格按照以下格式输出JSON:
{{"stance": "选择一个", "style": ["可以选择多个"]}}"""
    
    def analyze_stance(self, parent_comment: str, reply_comment: str) -> Tuple[str, List[str]]:
        """分析立场和互动风格（实时模式）"""
        try:
            payload = {
                "model": self.config.model,
                "messages": [
                    {"role": "user", "content": self.stance_prompt.format(
                        parent_comment=parent_comment,
                        reply_comment=reply_comment
                    )}
                ],
                "temperature": 0.1,
                "max_tokens": 100
            }
            
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.config.api_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_content = result["choices"][0]["message"]["content"].strip()
                return self._parse_response(raw_content)
            else:
                self.logger.warning(f"API请求失败: {response.status_code}")
                return self._get_default_values()
                
        except Exception as e:
            self.logger.error(f"立场分析失败: {str(e)}")
            return self._get_default_values()
    
    def _parse_response(self, content: str) -> Tuple[str, List[str]]:
        """解析LLM响应"""
        try:
            cleaned_content = content.strip()
            
            if cleaned_content.startswith('```json'):
                cleaned_content = cleaned_content[7:]
            if cleaned_content.startswith('```'):
                cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith('```'):
                cleaned_content = cleaned_content[:-3]
            
            cleaned_content = cleaned_content.strip()
            
            import json
            analysis = json.loads(cleaned_content)
            stance = analysis.get("stance", "中立/无关")
            styles = analysis.get("style", ["信息补充"])
            
            valid_stances = ["支持", "反对", "中立/无关"]
            if stance not in valid_stances:
                stance = "中立/无关"
            
            valid_styles = ["逻辑论证", "举例说明", "提出质疑", "幽默/玩梗", "信息补充"]
            styles = [style for style in styles if style in valid_styles]
            if not styles:
                styles = ["信息补充"]
            
            return stance, styles
            
        except Exception as e:
            self.logger.warning(f"JSON解析失败: {str(e)}")
            return self._get_default_values()
    
    @staticmethod
    def _get_default_values() -> Tuple[str, List[str]]:
        """获取默认值"""
        return "中立/无关", ["信息补充"]


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def calculate_simple_features(self, parent_comment: str, reply_comment: str) -> Dict[str, Union[int, float]]:
        """计算简化的语言风格特征"""
        features = {}
        
        parent_len = len(parent_comment)
        reply_len = len(reply_comment)
        features['length_ratio'] = reply_len / parent_len if parent_len > 0 else 0
        features['length_diff'] = reply_len - parent_len
        
        parent_ttr = self._calculate_ttr(parent_comment)
        reply_ttr = self._calculate_ttr(reply_comment)
        features['complexity_diff'] = reply_ttr - parent_ttr
        
        features.update(self._calculate_emoji_features(parent_comment, reply_comment))
        
        return features
    
    @staticmethod
    def _calculate_ttr(text: str) -> float:
        """计算类型-词例比(TTR)"""
        words = list(jieba.cut(text))
        if len(words) == 0:
            return 0
        unique_words = len(set(words))
        total_words = len(words)
        return unique_words / total_words
    
    @staticmethod
    def _calculate_emoji_features(parent: str, reply: str) -> Dict[str, int]:
        """计算表情符号特征"""
        parent_emojis = len(re.findall(r'\[.*?\]', parent))
        reply_emojis = len(re.findall(r'\[.*?\]', reply))
        return {
            'emoji_diff': reply_emojis - parent_emojis,
            'parent_emoji_count': parent_emojis,
            'reply_emoji_count': reply_emojis
        }
    
    def calculate_temporal_features(self, parent_time: str, reply_time: str) -> Dict[str, Union[int, float]]:
        """计算时间相关特征"""
        features = {}
        try:
            time_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M',
                '%Y-%m-%d %H:%M',
                '%Y/%m/%d %H:%M:%S',
                '%Y/%m/%d %H:%M',
            ]
            
            parent_dt = None
            reply_dt = None
            
            for fmt in time_formats:
                try:
                    parent_dt = pd.to_datetime(parent_time, format=fmt)
                    break
                except:
                    continue
            
            for fmt in time_formats:
                try:
                    reply_dt = pd.to_datetime(reply_time, format=fmt)
                    break
                except:
                    continue
            
            if parent_dt is None:
                parent_dt = pd.to_datetime(parent_time, errors='coerce')
            if reply_dt is None:
                reply_dt = pd.to_datetime(reply_time, errors='coerce')
            
            if parent_dt is not None and reply_dt is not None and not pd.isna(parent_dt) and not pd.isna(reply_dt):
                time_diff_hours = (reply_dt - parent_dt).total_seconds() / 3600
                features['time_diff_hours'] = time_diff_hours
                features['is_quick_reply'] = 1 if time_diff_hours <= 1 else 0
            else:
                self.logger.warning(f"时间解析失败: parent_time={parent_time}, reply_time={reply_time}")
                features['time_diff_hours'] = 0
                features['is_quick_reply'] = 0
            
        except Exception as e:
            self.logger.warning(f"时间特征计算失败: {str(e)}")
            features['time_diff_hours'] = 0
            features['is_quick_reply'] = 0
        
        return features


class CommentGraphAnalyzer:
    """评论图分析器 - 支持批量API"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger.setup_logger(level=config.log_level)
        
        # 初始化组件
        self.semantic_analyzer = SemanticAnalyzer(config.semantic_model_name, self.logger)
        self.stance_analyzer = LLMStanceAnalyzer(config, self.logger)
        self.feature_extractor = FeatureExtractor(self.logger)
        self.batch_manager = BatchAPIManager(config, self.logger) if config.use_batch_api else None
        
        # 特征缩放器
        self.scaler = StandardScaler()
        
        # 创建输出目录
        Path(self.config.output_folder).mkdir(parents=True, exist_ok=True)
    
    def extract_video_info(self, filename):
        """从文件名提取视频BV号和根评论ID"""
        pattern = r'评论树_([^_]+)_([^_]+)\.csv'
        match = re.search(pattern, filename)
        if match:
            return match.group(1), match.group(2)
        return None, None
    
    def load_comment_data(self, filepath):
        """加载评论数据CSV"""
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
            df['评论时间'] = pd.to_datetime(df['评论时间'])
            return df
        except Exception as e:
            self.logger.error(f"加载评论数据失败: {filepath}, 错误: {e}")
            return None
    
    def compute_node_features(self, comment_row):
        """计算单个节点的特征"""
        features = {}
        
        comment_text = str(comment_row['评论内容'])
        text_embedding = self.semantic_analyzer.encode_text(comment_text)
        features['text_embedding'] = text_embedding
        
        fans_count = float(comment_row['用户粉丝数']) if pd.notna(comment_row['用户粉丝数']) else 0.0
        features['user_fans_normalized'] = np.log1p(fans_count)
        
        features['likes_count'] = float(comment_row['点赞数']) if pd.notna(comment_row['点赞数']) else 0.0
        features['replies_count'] = float(comment_row['回复数']) if pd.notna(comment_row['回复数']) else 0.0
        
        comment_time = comment_row['评论时间']
        features['comment_age_days'] = (self.config.data_collection_date - comment_time).days
        features['video_age_when_commented'] = (comment_time - self.config.video_publish_date).days
        
        return features
    
    def compute_individual_subtree_stats(self, comment_data, reply_id):
        """计算某个回复的个体子树统计信息"""
        reply_row = comment_data[comment_data['评论ID'] == reply_id]
        if reply_row.empty:
            return 0.0, 0.0
        
        self_likes = float(reply_row.iloc[0]['点赞数']) if pd.notna(reply_row.iloc[0]['点赞数']) else 0.0
        
        children = comment_data[comment_data['上级评论ID'] == reply_id]
        
        if children.empty:
            return 0.0, self_likes
        
        total_replies = len(children)
        total_likes = self_likes + children['点赞数'].sum()
        
        for _, child in children.iterrows():
            child_id = child['评论ID']
            sub_replies, sub_likes = self.compute_individual_subtree_stats(comment_data, child_id)
            total_replies += sub_replies
            total_likes += sub_likes
        
        return float(total_replies), float(total_likes)
    
    def collect_interaction_pairs(self, comment_data):
        """收集所有互动对信息"""
        pairs = []
        comment_id_to_row = {}
        
        for idx, row in comment_data.iterrows():
            comment_id_to_row[row['评论ID']] = row
        
        for idx, row in comment_data.iterrows():
            comment_id = row['评论ID']
            parent_id = row['上级评论ID']
            
            if parent_id != 0 and parent_id in comment_id_to_row:
                parent_row = comment_id_to_row[parent_id]
                pairs.append({
                    'parent_row': parent_row,
                    'reply_row': row,
                    'parent_id': parent_id,
                    'reply_id': comment_id,
                    'parent_comment': str(parent_row['评论内容']),
                    'reply_comment': str(row['评论内容'])
                })
        
        return pairs
    
    def process_batch_stance_analysis(self, all_pairs):
        """批量处理立场分析"""
        self.logger.info(f"准备批量处理 {len(all_pairs)} 个互动对")
        
        # 创建批量请求
        requests = []
        for i, pair_info in enumerate(all_pairs):
            custom_id = f"{pair_info['file_source']}_{pair_info['parent_id']}_{pair_info['reply_id']}"
            request = self.batch_manager.create_batch_request(
                custom_id,
                pair_info['parent_comment'],
                pair_info['reply_comment']
            )
            requests.append(request)
            pair_info['custom_id'] = custom_id
        
        # 分批处理（如果超过批次大小限制）
        batch_results = {}
        
        for batch_num, i in enumerate(range(0, len(requests), self.config.batch_size)):
            batch_requests = requests[i:i + self.config.batch_size]
            self.logger.info(f"处理第 {batch_num + 1} 批，包含 {len(batch_requests)} 个请求")
            
            # 写入批量文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            input_file = os.path.join(self.config.batch_folder, f"batch_input_{timestamp}_{batch_num}.jsonl")
            self.batch_manager.write_batch_file(batch_requests, input_file)
            
            # 上传文件
            file_id = self.batch_manager.upload_file(input_file)
            self.logger.info(f"文件已上传，ID: {file_id}")
            
            # 创建批量任务
            batch_id = self.batch_manager.create_batch_job(file_id)
            self.logger.info(f"批量任务已创建，ID: {batch_id}")
            
            # 等待任务完成
            completed_batch = self.batch_manager.wait_for_batch_completion(batch_id)
            
            # 下载结果
            output_file = os.path.join(self.config.batch_folder, f"batch_output_{timestamp}_{batch_num}.jsonl")
            self.batch_manager.download_results(completed_batch.output_file_id, output_file)
            self.logger.info(f"结果已下载到: {output_file}")
            
            # 解析结果
            batch_stance_results = self.batch_manager.parse_batch_results(output_file)
            batch_results.update(batch_stance_results)
        
        # 将结果映射回原始数据
        for pair_info in all_pairs:
            custom_id = pair_info['custom_id']
            if custom_id in batch_results:
                pair_info['stance'], pair_info['styles'] = batch_results[custom_id]
            else:
                pair_info['stance'], pair_info['styles'] = "中立/无关", ["信息补充"]
        
        return all_pairs
    
    def analyze_interaction_pair(self, parent_row, reply_row, comment_data, stance=None, styles=None) -> Dict:
        """分析单个互动对的特征（立场信息可选）"""
        parent_comment = str(parent_row['评论内容'])
        reply_comment = str(reply_row['评论内容'])
        parent_id = parent_row['评论ID']
        reply_id = reply_row['评论ID']
        
        features = {}
        
        # 1. 如果没有提供立场信息，则实时分析
        if stance is None or styles is None:
            stance, styles = self.stance_analyzer.analyze_stance(parent_comment, reply_comment)
        
        # 立场特征
        features['stance_support'] = 1.0 if stance == "支持" else 0.0
        features['stance_oppose'] = 1.0 if stance == "反对" else 0.0
        features['stance_neutral'] = 1.0 if stance == "中立/无关" else 0.0
        
        # 风格特征
        features['style_logical'] = 1.0 if "逻辑论证" in styles else 0.0
        features['style_example'] = 1.0 if "举例说明" in styles else 0.0
        features['style_question'] = 1.0 if "提出质疑" in styles else 0.0
        features['style_humor'] = 1.0 if "幽默/玩梗" in styles else 0.0
        features['style_info'] = 1.0 if "信息补充" in styles else 0.0
        
        # 2. 语义相似度
        semantic_similarity = self.semantic_analyzer.calculate_semantic_similarity(
            parent_comment, reply_comment
        )
        features['semantic_similarity'] = semantic_similarity
        
        # 3. 语言风格特征
        stylistic_features = self.feature_extractor.calculate_simple_features(
            parent_comment, reply_comment
        )
        features.update(stylistic_features)
        
        # 4. 时间特征
        temporal_features = self.feature_extractor.calculate_temporal_features(
            parent_row['评论时间'], reply_row['评论时间']
        )
        features.update(temporal_features)
        
        # 5. 用户关系特征
        parent_fans = parent_row.get('用户粉丝数', 0)
        reply_fans = reply_row.get('用户粉丝数', 0)
        features['is_same_user'] = 1.0 if parent_row['用户ID'] == reply_row['用户ID'] else 0.0
        features['fans_ratio'] = reply_fans / parent_fans if parent_fans > 0 else 1.0
        
        # 6. 计算标签
        individual_subtree_replies, individual_subtree_likes = self.compute_individual_subtree_stats(
            comment_data, reply_id
        )
        
        hotness_score = individual_subtree_replies + self.config.alpha * individual_subtree_likes
        y_regression = np.log1p(hotness_score)
        
        labels = {
            'individual_subtree_replies': individual_subtree_replies,
            'individual_subtree_likes': individual_subtree_likes,
            'hotness_score': hotness_score,
            'y_regression': y_regression,
            'alpha': self.config.alpha
        }
        
        return features, labels
    
    def generate_ranking_pairs(self, comment_data, edge_labels):
        """为排序任务生成样本对"""
        ranking_pairs = []
        
        parent_groups = comment_data.groupby('上级评论ID')
        
        for parent_id, group in parent_groups:
            if parent_id == 0:
                continue
                
            replies = group['评论ID'].tolist()
            if len(replies) < 2:
                continue
            
            for i, reply_i in enumerate(replies):
                for j, reply_j in enumerate(replies):
                    if i == j:
                        continue
                    
                    edge_i = (reply_i, parent_id)
                    edge_j = (reply_j, parent_id)
                    
                    if edge_i in edge_labels and edge_j in edge_labels:
                        hotness_i = edge_labels[edge_i]['hotness_score']
                        hotness_j = edge_labels[edge_j]['hotness_score']
                        
                        ranking_label = 1 if hotness_i > hotness_j else 0
                        
                        ranking_pairs.append({
                            'reply_i': reply_i,
                            'reply_j': reply_j,
                            'parent_id': parent_id,
                            'hotness_i': hotness_i,
                            'hotness_j': hotness_j,
                            'label': ranking_label,
                            'hotness_diff': hotness_i - hotness_j
                        })
        
        return ranking_pairs
    
    def build_comment_tree_with_batch(self, comment_data, root_comment_id, file_source, stance_results=None):
        """构建子图（支持批量立场结果）"""
        G = nx.DiGraph()
        
        # 添加节点
        node_features = {}
        for idx, row in comment_data.iterrows():
            comment_id = row['评论ID']
            features = self.compute_node_features(row)
            node_features[comment_id] = features
            G.add_node(comment_id, **features)
        
        # 添加边
        edge_features = {}
        edge_labels = {}
        
        comment_id_to_row = {}
        for idx, row in comment_data.iterrows():
            comment_id_to_row[row['评论ID']] = row
        
        for idx, row in comment_data.iterrows():
            comment_id = row['评论ID']
            parent_id = row['上级评论ID']
            
            if parent_id != 0 and parent_id in comment_id_to_row:
                parent_row = comment_id_to_row[parent_id]
                
                # 如果有批量结果，使用批量结果
                stance = None
                styles = None
                if stance_results:
                    key = f"{file_source}_{parent_id}_{comment_id}"
                    if key in stance_results:
                        stance = stance_results[key]['stance']
                        styles = stance_results[key]['styles']
                
                # 分析互动对
                edge_feat, edge_lab = self.analyze_interaction_pair(
                    parent_row, row, comment_data, stance, styles
                )
                
                G.add_edge(comment_id, parent_id)
                edge_features[(comment_id, parent_id)] = edge_feat
                edge_labels[(comment_id, parent_id)] = edge_lab
                
                # 如果不使用批量API，添加延迟
                if not self.config.use_batch_api:
                    time.sleep(self.config.api_delay)
        
        # 生成排序对
        ranking_pairs = self.generate_ranking_pairs(comment_data, edge_labels)
        
        # 统计信息
        regression_stats = {
            'total_edges': len(edge_labels),
            'avg_individual_replies': np.mean([labels['individual_subtree_replies'] for labels in edge_labels.values()]) if edge_labels else 0,
            'avg_individual_likes': np.mean([labels['individual_subtree_likes'] for labels in edge_labels.values()]) if edge_labels else 0,
            'avg_hotness_score': np.mean([labels['hotness_score'] for labels in edge_labels.values()]) if edge_labels else 0,
            'avg_y_regression': np.mean([labels['y_regression'] for labels in edge_labels.values()]) if edge_labels else 0,
            'max_hotness_score': max([labels['hotness_score'] for labels in edge_labels.values()]) if edge_labels else 0,
            'min_hotness_score': min([labels['hotness_score'] for labels in edge_labels.values()]) if edge_labels else 0
        }
        
        ranking_stats = {
            'total_pairs': len(ranking_pairs),
            'positive_pairs': sum(1 for pair in ranking_pairs if pair['label'] == 1),
            'negative_pairs': sum(1 for pair in ranking_pairs if pair['label'] == 0),
            'balance_ratio': sum(1 for pair in ranking_pairs if pair['label'] == 1) / len(ranking_pairs) if ranking_pairs else 0
        }
        
        graph_data = {
            'graph': G,
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_labels': edge_labels,
            'ranking_pairs': ranking_pairs,
            'regression_stats': regression_stats,
            'ranking_stats': ranking_stats,
            'root_comment_id': root_comment_id,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'task_config': {
                'alpha': self.config.alpha,
                'regression_target': 'y_regression',
                'ranking_enabled': True
            }
        }
        
        return graph_data
    
    def process_batch(self) -> None:
        """批量处理CSV文件"""
        csv_files = [f for f in os.listdir(self.config.input_folder) if f.endswith('.csv')]
        csv_files.sort()
        
        if not csv_files:
            self.logger.error(f"在 {self.config.input_folder} 中未找到CSV文件")
            return
        
        total_files = len(csv_files)
        start_idx = self.config.start_index
        end_idx = min(self.config.end_index, total_files)
        
        if start_idx >= total_files:
            self.logger.error(f"起始索引 {start_idx} 超过了文件总数 {total_files}")
            return
        
        files_to_process = csv_files[start_idx:end_idx]
        
        self.logger.info(f"找到 {total_files} 个CSV文件")
        self.logger.info(f"将处理第 {start_idx + 1} 到第 {end_idx} 个文件，共 {len(files_to_process)} 个文件")
        
        if self.config.use_batch_api:
            # 批量模式：先收集所有互动对，批量分析立场，再构建图
            all_pairs = []
            file_data_map = {}
            
            # 收集所有互动对
            for csv_file in files_to_process:
                filepath = os.path.join(self.config.input_folder, csv_file)
                bv_id, root_comment_id = self.extract_video_info(csv_file)
                
                if not bv_id or not root_comment_id:
                    continue
                
                comment_data = self.load_comment_data(filepath)
                if comment_data is None:
                    continue
                
                file_source = f"{bv_id}_{root_comment_id}"
                pairs = self.collect_interaction_pairs(comment_data)
                
                for pair in pairs:
                    pair['file_source'] = file_source
                    all_pairs.append(pair)
                
                file_data_map[csv_file] = {
                    'bv_id': bv_id,
                    'root_comment_id': root_comment_id,
                    'comment_data': comment_data,
                    'file_source': file_source
                }
            
            # 批量分析立场
            if all_pairs:
                all_pairs = self.process_batch_stance_analysis(all_pairs)
                
                # 整理立场结果
                stance_results = {}
                for pair in all_pairs:
                    key = pair['custom_id']
                    stance_results[key] = {
                        'stance': pair['stance'],
                        'styles': pair['styles']
                    }
                
                # 构建图
                success_count = 0
                for csv_file, file_info in file_data_map.items():
                    try:
                        graph_data = self.build_comment_tree_with_batch(
                            file_info['comment_data'],
                            int(file_info['root_comment_id']),
                            file_info['file_source'],
                            stance_results
                        )
                        
                        # 保存子图
                        output_filename = f"subgraph_{file_info['bv_id']}_{file_info['root_comment_id']}.pkl"
                        output_filepath = os.path.join(self.config.output_folder, output_filename)
                        
                        with open(output_filepath, 'wb') as f:
                            pickle.dump(graph_data, f)
                        
                        self.logger.info(f"✅ 子图已保存: {output_filename}")
                        success_count += 1
                        
                    except Exception as e:
                        self.logger.error(f"❌ 构建子图失败 {csv_file}: {e}")
            
        else:
            # 实时模式：逐个处理
            success_count = 0
            for i, csv_file in enumerate(tqdm(files_to_process, desc="构建图")):
                csv_path = os.path.join(self.config.input_folder, csv_file)
                actual_file_number = start_idx + i + 1
                self.logger.info(f"\n处理第 {actual_file_number}/{total_files} 个文件: {csv_file}")
                
                if self.process_single_file(csv_path):
                    success_count += 1
        
        # 生成报告
        self._generate_processing_report(len(files_to_process), success_count)
        self.logger.info("批量处理完成")
    
    def process_single_file(self, filepath):
        """处理单个文件（实时模式）"""
        filename = os.path.basename(filepath)
        bv_id, root_comment_id = self.extract_video_info(filename)
        
        if not bv_id or not root_comment_id:
            self.logger.error(f"无法解析文件名: {filename}")
            return False
        
        self.logger.info(f"处理文件: {filename}")
        self.logger.info(f"  BV号: {bv_id}, 根评论ID: {root_comment_id}")
        
        comment_data = self.load_comment_data(filepath)
        if comment_data is None:
            return False
        
        try:
            file_source = f"{bv_id}_{root_comment_id}"
            graph_data = self.build_comment_tree_with_batch(
                comment_data, int(root_comment_id), file_source, None
            )
            
            output_filename = f"subgraph_{bv_id}_{root_comment_id}.pkl"
            output_filepath = os.path.join(self.config.output_folder, output_filename)
            
            with open(output_filepath, 'wb') as f:
                pickle.dump(graph_data, f)
            
            self.logger.info(f"  ✅ 子图已保存: {output_filename}")
            self.logger.info(f"     节点数: {graph_data['num_nodes']}, 边数: {graph_data['num_edges']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"  ❌ 构建子图失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_processing_report(self, total_processed: int, success_count: int) -> None:
        """生成处理报告"""
        report_path = os.path.join(
            self.config.output_folder, 
            f"graph_construction_report_files_{self.config.start_index + 1}_to_{self.config.end_index}.txt"
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== 图构建处理报告 ===\n\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理模式: {'批量API' if self.config.use_batch_api else '实时API'}\n")
            f.write(f"处理文件范围: 第 {self.config.start_index + 1} 到第 {self.config.end_index} 个文件\n\n")
            
            f.write(f"总处理文件数: {total_processed}\n")
            f.write(f"成功构建图数: {success_count}\n")
            f.write(f"失败文件数: {total_processed - success_count}\n")
            f.write(f"成功率: {success_count/total_processed*100:.1f}%\n\n")
            
            f.write("=== 配置信息 ===\n")
            f.write(f"输入文件夹: {self.config.input_folder}\n")
            f.write(f"输出文件夹: {self.config.output_folder}\n")
            f.write(f"API模型: {self.config.model}\n")
            f.write(f"语义模型: {self.config.semantic_model_name}\n")
            f.write(f"点赞权重系数(α): {self.config.alpha}\n")
            f.write(f"视频发布日期: {self.config.video_publish_date}\n")
            f.write(f"数据收集日期: {self.config.data_collection_date}\n")
            
            if self.config.use_batch_api:
                f.write(f"\n=== 批量API配置 ===\n")
                f.write(f"批次大小: {self.config.batch_size}\n")
                f.write(f"临时文件夹: {self.config.batch_folder}\n")
        
        self.logger.info(f"处理报告已保存到: {report_path}")


def main():
    """主函数"""
    try:
        # 初始化配置
        config = Config()
        
        # 可以通过设置选择使用批量API还是实时API
        # config.use_batch_api = True  # 使用批量API（费用减半）
        # config.use_batch_api = False  # 使用实时API
        
        # 创建分析器并运行
        analyzer = CommentGraphAnalyzer(config)
        analyzer.process_batch()
        
        print("\n=== 处理完成 ===")
        print(f"结果保存在: {config.output_folder}")
        print(f"处理文件范围: 第 {config.start_index + 1} 到第 {config.end_index} 个文件")
        print(f"处理模式: {'批量API（费用减半）' if config.use_batch_api else '实时API'}")
        print("\n生成的文件包括:")
        print("- 各评论树的子图文件（.pkl格式）")
        print("- 图构建处理报告")
        
        if config.use_batch_api:
            print(f"\n批量处理临时文件保存在: {config.batch_folder}")
            print("包含输入JSONL文件和输出结果文件")
        
    except Exception as e:
        print(f"程序运行失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()