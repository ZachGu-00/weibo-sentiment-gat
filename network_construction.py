import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
from datetime import datetime
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
import re

class UserInteractionNetwork:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.df = None
        self.G = nx.DiGraph()  

        self.unique_posts = None
        self.unique_comments = None
        self.unique_second_comments = None

        self.user_texts = defaultdict(list)
        self.user_times = defaultdict(list)

        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.model = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.model.eval()
        print("RoBERTa is ready")

    def load_data(self):
        self.df = pd.read_excel(self.excel_path)
        self.df = self.df.dropna(subset=['发帖人'])
        self.df['博文内容'] = self.df['博文内容'].fillna('')
        self.df['评论内容'] = self.df['评论内容'].fillna('')
        self.df['二级评论内容'] = self.df['二级评论内容'].fillna('')
        return self.df

    def deduplicate_data(self):
        """数据去重 - 解决核心问题"""
        print("=== 数据去重处理 ===")

        # 1. 去重发帖记录：按照发帖人+博文内容+发帖时间去重
        print("去重发帖记录...")
        posts_columns = ['博文网址', '发帖人', '博文发布时间', '博文内容', '转发数', '评论数', '点赞数']
        available_posts_columns = [col for col in posts_columns if col in self.df.columns]

        self.unique_posts = self.df[available_posts_columns].drop_duplicates(
            subset=['发帖人', '博文内容', '博文发布时间']
        ).dropna(subset=['发帖人', '博文内容'])

        print(f"去重前：{len(self.df)}行 -> 去重后发帖记录：{len(self.unique_posts)}条")

        # 2. 去重评论记录：按照发帖人+博文内容+评论人+评论内容去重
        print("去重评论记录...")
        comment_data = self.df[['发帖人', '博文内容', '评论人', '评论内容', '评论时间', '评论获赞']].dropna(
            subset=['发帖人', '博文内容', '评论人', '评论内容']
        )

        self.unique_comments = comment_data.drop_duplicates(
            subset=['发帖人', '博文内容', '评论人', '评论内容']
        )

        print(f"去重后评论记录：{len(self.unique_comments)}条")

        # 3. 去重二级评论记录：按照评论人+评论内容+二级评论人+二级评论内容去重
        print("去重二级评论记录...")
        second_comment_data = self.df[['评论人', '评论内容', '二级评论人', '二级评论内容', '二级评论时间']].dropna(
            subset=['评论人', '评论内容', '二级评论人', '二级评论内容']
        )

        self.unique_second_comments = second_comment_data.drop_duplicates(
            subset=['评论人', '评论内容', '二级评论人', '二级评论内容']
        )

        print(f"去重后二级评论记录：{len(self.unique_second_comments)}条")

        print("数据去重完成")

    def preprocess_time(self, time_str):
        """将时间处理到日期级别"""
        if pd.isna(time_str):
            return None
        try:
            dt = datetime.strptime(str(time_str), '%Y-%m-%d %H:%M:%S')
            return dt.strftime('%Y-%m-%d')
        except:
            try:
                dt = datetime.strptime(str(time_str), '%Y-%m-%d')
                return dt.strftime('%Y-%m-%d')
            except:
                return None

    def clean_text(self, text):
        """清理文本"""
        if pd.isna(text) or text == '':
            return ''

        text = str(text)
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # 去除@用户名
        text = re.sub(r'@[\\w]+', '', text)
        # 去除多余空格
        text = re.sub(r'\\s+', ' ', text).strip()

        return text

    def get_roberta_embeddings(self, text):
        if not text or len(text.strip()) == 0:
            return np.zeros(768)

        try:
            cleaned_text = self.clean_text(text)
            if len(cleaned_text.strip()) == 0:
                return np.zeros(768)

            inputs = self.tokenizer(cleaned_text,
                                  return_tensors='pt',
                                  max_length=512,
                                  truncation=True,
                                  padding=True)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()

            return embeddings.flatten()

        except Exception as e:
            return np.zeros(768)

    def collect_user_data_from_deduplicated(self):

        # 1. 从发帖记录收集
        print("收集发帖数据...")
        for idx, row in self.unique_posts.iterrows():
            user = row['发帖人']
            content = self.clean_text(row['博文内容'])
            time = self.preprocess_time(row['博文发布时间'])

            if content:
                self.user_texts[user].append(content)
                if time:
                    self.user_times[user].append(time)

        # 2. 从评论记录收集
        print("收集评论数据...")
        for idx, row in self.unique_comments.iterrows():
            user = row['评论人']
            content = self.clean_text(row['评论内容'])
            time = self.preprocess_time(row['评论时间'])

            if content:
                self.user_texts[user].append(content)
                if time:
                    self.user_times[user].append(time)

        # 3. 从二级评论记录收集
        print("收集二级评论数据...")
        for idx, row in self.unique_second_comments.iterrows():
            user = row['二级评论人']
            content = self.clean_text(row['二级评论内容'])
            time = self.preprocess_time(row['二级评论时间'])

            if content:
                self.user_texts[user].append(content)
                if time:
                    self.user_times[user].append(time)

        print(f"收集到 {len(self.user_texts)} 个用户的数据")

        user_post_counts = [(user, len(texts)) for user, texts in self.user_texts.items()]
        user_post_counts.sort(key=lambda x: x[1], reverse=True)



    def build_network(self):
        self.deduplicate_data()
        self.collect_user_data_from_deduplicated()
        for idx, user in enumerate(self.user_texts.keys()):
            if idx % 50 == 0:
                print(f"Procedure: {idx}/{len(self.user_texts)}")

            user_texts = self.user_texts[user]
            individual_embeddings = []
            for text in user_texts:
                if text.strip():
                    embedding = self.get_roberta_embeddings(text)
                    individual_embeddings.append(embedding)
                    
            # 平均池化
            if individual_embeddings:
                avg_embeddings = np.mean(individual_embeddings, axis=0)
                individual_embeddings_array = np.array(individual_embeddings)
            else:
                avg_embeddings = np.zeros(768)
                individual_embeddings_array = np.array([]).reshape(0, 768)

   
            user_times = sorted(self.user_times.get(user, []))
            first_time = user_times[0] if user_times else None
            last_time = user_times[-1] if user_times else None

            # 这部分是节点特征
            self.G.add_node(user,
                          user_name=user,
                          embeddings=avg_embeddings, 
                          individual_embeddings=individual_embeddings_array, 
                          num_posts_embedded=len(individual_embeddings), 
                          first_activity_date=first_time,
                          last_activity_date=last_time,
                          total_posts=len(self.user_texts[user]),
                          activity_days=len(set(user_times)) if user_times else 0)



        # 互动边1：评论人 -> 发帖人（基于去重后的评论记录）
        comment_interactions = defaultdict(int)
        for idx, row in self.unique_comments.iterrows():
            poster = row['发帖人']
            commenter = row['评论人']

            if poster != commenter and poster in self.G.nodes and commenter in self.G.nodes:
                comment_interactions[(commenter, poster)] += 1

        # 添加评论互动边
        for (commenter, poster), weight in comment_interactions.items():
            self.G.add_edge(commenter, poster,
                          weight=weight,
                          interaction_type='comment')


        # 互动边2：二级评论人 -> 评论人
        reply_interactions = defaultdict(int)
        for idx, row in self.unique_second_comments.iterrows():
            commenter = row['评论人']
            second_commenter = row['二级评论人']

            if commenter != second_commenter and commenter in self.G.nodes and second_commenter in self.G.nodes:
                reply_interactions[(second_commenter, commenter)] += 1

        # 添加回复互动边
        for (second_commenter, commenter), weight in reply_interactions.items():
            self.G.add_edge(second_commenter, commenter,
                          weight=weight,
                          interaction_type='reply')



  

    def save_network(self, filename_prefix="user_interaction_network"):

        with open(f"{filename_prefix}.pkl", 'wb') as f:
            pickle.dump(self.G, f)
        print(f"网络已保存为: {filename_prefix}.pkl")

        # 保存去重后的数据表
        with pd.ExcelWriter(f"{filename_prefix}_deduplicated_data.xlsx") as writer:
            self.unique_posts.to_excel(writer, sheet_name='发帖记录', index=False)
            self.unique_comments.to_excel(writer, sheet_name='评论记录', index=False)
            self.unique_second_comments.to_excel(writer, sheet_name='二级评论记录', index=False)


        G_copy = self.G.copy()
        for node in G_copy.nodes():
            embeddings = G_copy.nodes[node]['embeddings']
            G_copy.nodes[node]['embeddings_str'] = ','.join(map(str, embeddings))
            del G_copy.nodes[node]['embeddings']
            del G_copy.nodes[node]['individual_embeddings']  

        nx.write_graphml(G_copy, f"{filename_prefix}.graphml")

        node_data = []
        for node, attrs in self.G.nodes(data=True):
            node_info = {
                'user_name': attrs['user_name'],
                'first_activity_date': attrs['first_activity_date'],
                'last_activity_date': attrs['last_activity_date'],
                'total_posts': attrs['total_posts'],
                'num_posts_embedded': attrs['num_posts_embedded'],
                'activity_days': attrs['activity_days'],
                'in_degree': self.G.in_degree(node),
                'out_degree': self.G.out_degree(node),
                'total_degree': self.G.degree(node)
            }
            node_data.append(node_info)

        node_df = pd.DataFrame(node_data)
        node_df.to_csv(f"{filename_prefix}_nodes.csv", index=False, encoding='utf-8-sig')

        edge_data = []
        for u, v, attrs in self.G.edges(data=True):
            edge_info = {
                'source': u,
                'target': v,
                'weight': attrs['weight'],
                'interaction_type': attrs['interaction_type']
            }
            edge_data.append(edge_info)

        edge_df = pd.DataFrame(edge_data)
        edge_df.to_csv(f"{filename_prefix}_edges.csv", index=False, encoding='utf-8-sig')


def main():
    excel_path = "/content/drive/MyDrive/印巴/印巴_comment_weibo.xlsx"


    network_builder = UserInteractionNetwork(excel_path)
    network_builder.load_data()
    network_builder.build_network()
    network_builder.print_network_statistics()
    network_builder.save_network("weibo_user_interaction_network_deduped")

    return network_builder.G

if __name__ == "__main__":
    network = main()