import torch
import torch_musa
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
import json
from sklearn.model_selection import train_test_split
import os
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm

# 设置中文字体支持
matplotlib.use('Agg')  # 避免需要GUI环境

class GraphQueryDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 构建类型和关系的词汇表
        self.node_types = {"<pad>": 0, "<unk>": 1}
        self.relations = {"<pad>": 0, "<unk>": 1}
        
        type_idx = 2
        rel_idx = 2
        
        for example in examples:
            graph = example["graph"]
            
            # 处理节点类型
            for node_id, node_info in graph["nodes"].items():
                node_type = node_info["type"]
                if node_type not in self.node_types:
                    self.node_types[node_type] = type_idx
                    type_idx += 1
            
            # 处理关系类型
            for edge in graph["edges"]:
                relation = edge["relation"]
                if relation not in self.relations:
                    self.relations[relation] = rel_idx
                    rel_idx += 1
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        question = example["question"]
        graph = example["graph"]
        
        # 获取主实体类型（通常是n0）
        main_entity_type = graph["nodes"]["n0"]["type"]
        main_entity_value = graph["nodes"]["n0"]["value"]
        
        # 获取查询节点的类型
        query_node = graph["query_node"]
        query_type = graph["nodes"][query_node]["type"]
        
        # 获取关系（通常是从主实体到查询节点的边）
        relation = None
        for edge in graph["edges"]:
            if edge["to"] == query_node:
                relation = edge["relation"]
                break
        
        # 编码文本
        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 转换为所需的ID
        main_entity_type_id = self.node_types.get(main_entity_type, 1)  # 1是<unk>
        relation_id = self.relations.get(relation, 1) if relation else 1
        query_type_id = self.node_types.get(query_type, 1)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "main_entity_type": torch.tensor(main_entity_type_id),
            "relation": torch.tensor(relation_id),
            "query_type": torch.tensor(query_type_id),
            "question": question,
            "main_entity": main_entity_value
        }

class GraphQueryParser(nn.Module):
    def __init__(self, pretrained_model="hfl/chinese-roberta-wwm-ext", hidden_dim=768, 
                 num_node_types=50, num_relations=30, dropout_rate=0.2):
        super().__init__()
        # 使用更大/更适合的预训练模型
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        encoder_dim = self.encoder.config.hidden_size
        
        # 深层转换网络
        self.transform = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(dropout_rate),
            nn.GELU(),  # GELU比ReLU在很多NLP任务上表现更好
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.GELU()
        )
        
        # 自定义注意力实现 (不使用nn.MultiheadAttention)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 创建关系识别专用的注意力机制
        self.relation_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.relation_key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.relation_value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 关系特定的池化层
        self.relation_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 主实体预测器 - 不变
        self.main_entity_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_dim//2, num_node_types)
        )
        
        # 关系预测器 - 独立路径，不依赖实体类型
        self.relation_decoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),  # 使用句子表示和特定关系表示
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_dim//2, num_relations)
        )
        
        # 查询类型预测器 - 使用多模态融合
        self.query_type_fusion = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),  # 融合三种表示
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.GELU()
        )
        
        self.query_type_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_dim//2, num_node_types)
        )
        
        # 修改: 调整兼容性矩阵的维度为[num_node_types, num_relations]
        self.compatibility_matrix = nn.Parameter(
            torch.zeros(num_node_types, num_relations)
        )
        
        # 添加兼容性投影层作为替代
        self.compatibility_projection = nn.Linear(num_node_types, hidden_dim)
    
    def custom_attention(self, query, key, value, attn_mask=None):
        """自定义注意力实现，避免使用nn.MultiheadAttention"""
        # 投影查询、键、值
        q = self.query_proj(query)  # [batch_size, seq_len, hidden_dim]
        k = self.key_proj(key)      # [batch_size, seq_len, hidden_dim]
        v = self.value_proj(value)  # [batch_size, seq_len, hidden_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32, device=k.device))
        
        # 应用掩码（如果提供）
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), -1e9)
        
        # 应用softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 加权组合值向量
        context = torch.matmul(attn_weights, v)
        
        # 输出投影
        output = self.out_proj(context)
        
        return output
    
    def forward(self, input_ids, attention_mask):
        # 编码文本
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取每个token的表示
        token_repr = outputs.last_hidden_state
        
        # 应用自定义注意力 (替代之前的self.attention调用)
        bool_mask = attention_mask.bool()
        attn_output = self.custom_attention(token_repr, token_repr, token_repr, bool_mask)
        
        # 残差连接
        enhanced_repr = token_repr + attn_output
        
        # 使用[CLS]令牌表示
        sentence_repr = enhanced_repr[:, 0, :]
        
        # 转换表示
        transformed = self.transform(sentence_repr)
        
        # 1. 预测主实体类型（与原来相同）
        main_entity_logits = self.main_entity_decoder(transformed)
        main_entity_probs = F.softmax(main_entity_logits, dim=1)
        
        # 2. 创建关系特定的表示
        # 应用关系特定的注意力，识别句子中的关系指示词
        relation_q = self.relation_query_proj(transformed).unsqueeze(1)  # [B, 1, H]
        relation_k = self.relation_key_proj(enhanced_repr)  # [B, L, H]
        relation_v = self.relation_value_proj(enhanced_repr)  # [B, L, H]
        
        # 计算关系注意力分数
        relation_scores = torch.matmul(relation_q, relation_k.transpose(-2, -1)) / math.sqrt(relation_k.size(-1))
        if bool_mask is not None:
            relation_scores = relation_scores.masked_fill(~bool_mask.unsqueeze(1), -1e9)
        
        relation_attn_weights = F.softmax(relation_scores, dim=-1)
        relation_context = torch.matmul(relation_attn_weights, relation_v).squeeze(1)  # [B, H]
        
        # 增强关系表示
        relation_repr = self.relation_pooling(relation_context)
        
        # 独立预测关系（不使用主实体类型信息）
        relation_input = torch.cat([transformed, relation_repr], dim=1)
        relation_logits = self.relation_decoder(relation_input)
        relation_probs = F.softmax(relation_logits, dim=1)
        
        # 3. 预测查询类型，使用多模态融合
        # 修正: 使用兼容性投影层替代矩阵乘法
        entity_compat_repr = self.compatibility_projection(main_entity_probs)
        
        # 融合句子表示、关系表示和主实体表示
        query_fusion_input = torch.cat([transformed, relation_repr, entity_compat_repr], dim=1)
        query_fusion = self.query_type_fusion(query_fusion_input)
        query_type_logits = self.query_type_decoder(query_fusion)
        
        return main_entity_logits, relation_logits, query_type_logits
    
    def predict_query_graph(self, text, tokenizer, node_types, relations, device):
        """解析文本为查询图结构"""
        # 创建ID到类型/关系的映射
        id2type = {v: k for k, v in node_types.items()}
        id2relation = {v: k for k, v in relations.items()}
        
        # 编码输入
        encoding = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # 获取预测
        self.eval()
        with torch.no_grad():
            main_entity_logits, relation_logits, query_type_logits = self(input_ids, attention_mask)
            
            main_entity_type_id = torch.argmax(main_entity_logits, dim=1).item()
            relation_id = torch.argmax(relation_logits, dim=1).item()
            query_type_id = torch.argmax(query_type_logits, dim=1).item()
        
        # 转换为标签
        main_entity_type = id2type.get(main_entity_type_id, "<unk>")
        relation = id2relation.get(relation_id, "<unk>")
        query_type = id2type.get(query_type_id, "<unk>")
        
        # 从文本中提取主实体
        main_entity = self._extract_entity(text)
        
        # 创建查询图
        query_graph = {
            "nodes": {
                "n0": {"type": main_entity_type, "value": main_entity},
                "n1": {"type": query_type, "value": None}
            },
            "edges": [
                {"from": "n0", "to": "n1", "relation": relation}
            ],
            "query_node": "n1"
        }
        
        return query_graph
    
    def _extract_entity(self, text):
        """从文本中简单提取实体名称"""
        # 这是一个简化版本，实际应用中可能需要更复杂的NER
        # 假设实体通常出现在问题的前半部分，在特定符号前
        for delimiter in ["的", "是", "有", "在", "？", "?"]:
            if delimiter in text:
                parts = text.split(delimiter, 1)
                entity = parts[0].strip()
                # 过滤常见的问题引导词
                for prefix in ["请问", "告诉我", "我想知道", "请告诉我"]:
                    if entity.startswith(prefix):
                        entity = entity[len(prefix):].strip()
                return entity
        
        # 如果找不到明确的分隔符，返回前几个字符
        return text[:5]  # 简单截取前5个字符

class GraphQuerySystem:
    def __init__(self, model, tokenizer, device, config_path="data/graph_parser/common/relations_config.json"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.field_mapping = {rel: info["db_field"] for rel, info in self.config["relations"].items()}
        self.type_mapping = {type_name: type_info.get("template_key", type_name) 
                            for type_name, type_info in self.config["node_types"].items()}
        self.relation_to_query_type = {rel: info["compatible_types"] for rel, info in self.config["relations"].items()}
        
        # 从配置文件构建关键词到关系的映射
        self.keyword_to_relation = {}
        for relation, config in self.config["relations"].items():
            if "keywords" in config:
                for keyword in config["keywords"]:
                    self.keyword_to_relation[keyword] = relation
        
        # 加载知识库
        self.db = self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """从JSON文件加载大学知识库"""
        try:
            knowledge_base_path = "data/graph_parser/common/university_knowledge_base.json"
            print(f"加载知识库: {knowledge_base_path}")
            with open(knowledge_base_path, "r", encoding="utf-8") as f:
                knowledge_base = json.load(f)
                print(f"成功加载知识库，包含{len(knowledge_base)}所大学的信息")
                return knowledge_base
        except Exception as e:
            print(f"加载知识库时出错: {e}")
            # 如果无法加载，返回空字典或基本示例
            return {}
    
    def predict_query_graph(self, text):
        """使用模型解析文本为查询图结构，并进行逻辑约束验证"""
        # 获取模型预测的查询图
        query_graph = self.model.predict_query_graph(text, self.tokenizer, self.config["node_types"], self.config["relations"], self.device)
        
        # 提取当前预测的关系和查询类型
        relation = None
        query_node = query_graph["query_node"]
        query_type = query_graph["nodes"][query_node]["type"]
        
        for edge in query_graph["edges"]:
            if edge["to"] == query_node:
                relation = edge["relation"]
                break
        
        # 应用关系-查询类型兼容性验证
        if relation and query_type:
            # 获取与当前关系兼容的查询类型列表
            compatible_query_types = self.relation_to_query_type.get(relation, [])
            
            # 如果预测的查询类型与关系不兼容，进行修正
            if query_type not in compatible_query_types and compatible_query_types:
                # 使用第一个兼容的查询类型替换不兼容的查询类型
                query_graph["nodes"][query_node]["type"] = compatible_query_types[0]
                
                # 输出纠正信息（仅调试用）
                # print(f"修正查询类型: {query_type} → {compatible_query_types[0]} (关系: {relation})")
        
        # 使用规则修正关系（基于关键词）
        relation = self.correct_relation_by_keywords(text, relation)
        for edge in query_graph["edges"]:
            if edge["to"] == query_node:
                edge["relation"] = relation
        
        return query_graph
    
    def correct_relation_by_keywords(self, text, current_relation):
        """基于文本关键词修正关系预测"""
        # 使用从配置文件加载的关键词映射
        for keyword, relation in self.keyword_to_relation.items():
            if keyword in text:
                return relation
        
        # 如果没有找到匹配的关键词，返回原关系
        return current_relation
    
    def query_database(self, query_graph):
        """查询数据库获取缺失信息"""
        main_node = query_graph["nodes"]["n0"]
        query_node = query_graph["nodes"][query_graph["query_node"]]
        main_value = main_node["value"]
        
        if main_value not in self.db:
            return None
        
        # 从边获取属性名
        for edge in query_graph["edges"]:
            if edge["to"] == query_graph["query_node"]:
                relation = edge["relation"]
                
                # 处理设施相关查询
                if relation == "has_facility":
                    # 查找连接到设施的节点
                    facility_node_id = edge["to"]
                    facility_type = query_graph["nodes"][facility_node_id]["type"]
                    
                    # 检查是否有设施信息
                    if "facilities" in self.db[main_value] and facility_type in self.db[main_value]["facilities"]:
                        facility_info = self.db[main_value]["facilities"][facility_type]
                        
                        # 检查是否需要查询设施的位置
                        for other_edge in query_graph["edges"]:
                            if other_edge["from"] == facility_node_id and other_edge["relation"] == "located_at":
                                return facility_info.get("location", "未知位置")
                        
                        # 默认返回设施名称
                        return facility_info.get("name", "未知设施")
                    
                    return "未找到相关设施信息"
                
                # 普通关系查询
                field = self.field_mapping.get(relation)
                if field and field in self.db[main_value]:
                    return self.db[main_value][field]
        
        return None
    
    def generate_response(self, query_graph, result):
        """根据查询图和结果生成自然语言回答"""
        main_node = query_graph["nodes"]["n0"]
        query_node = query_graph["nodes"][query_graph["query_node"]]
        
        main_value = main_node["value"]
        query_type = query_node["type"]
        
        # 从边获取关系
        relation = None
        for edge in query_graph["edges"]:
            if edge["to"] == query_graph["query_node"]:
                relation = edge["relation"]
                break
                
        # 获取关系的显示名称（如果存在）
        relation_display = relation
        if relation in self.config["relations"]:
            relation_display = self.config["relations"][relation].get("display_name", relation)
                
        # 使用模板生成回答
        template_key = self.type_mapping.get(query_type)
        if template_key and template_key in self.config["templates"]:
            # 如果模板中包含relation_display占位符，传入关系显示名称
            template = self.config["templates"][template_key] 
            if "{relation_display}" in template:
                return template.format(entity=main_value, value=result, relation_display=relation_display)
            else:
                return template.format(entity=main_value, value=result)
        
        # 默认回复
        return f"{main_value}的{relation_display}是{result}。"
    
    def visualize_graph(self, graph):
        """以文本形式可视化查询图"""
        print("\n====== 查询图结构 ======\n")
        
        print("节点:")
        for node_id, node_info in graph["nodes"].items():
            value_str = node_info["value"] if node_info["value"] else "待查询"
            query_marker = " [查询节点]" if node_id == graph["query_node"] else ""
            print(f"  {node_id}{query_marker}: 类型={node_info['type']}, 值={value_str}")
        
        print("\n关系:")
        for edge in graph["edges"]:
            from_node = graph["nodes"][edge["from"]]
            to_node = graph["nodes"][edge["to"]]
            from_value = from_node["value"] if from_node["value"] else "待查询"
            to_value = to_node["value"] if to_node["value"] else "待查询"
            print(f"  {edge['from']}({from_node['type']}:{from_value}) --[{edge['relation']}]--> {edge['to']}({to_node['type']}:{to_value})")
        
        print("\n======================\n")
    
    def visualize_graph_image(self, graph, output_path="query_graph.png"):
        """
        将查询图绘制为图像并保存
        
        参数:
            graph: 查询图数据结构
            output_path: 输出图像的文件路径
        """
        # 设置中文字体支持
        import matplotlib.font_manager as fm
        
        # 直接使用AR PL UMing CN字体
        plt.rcParams['font.sans-serif'] = ['AR PL UMing CN'] + plt.rcParams['font.sans-serif']
        print("使用中文字体: AR PL UMing CN")
        
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # 创建有向图
        G = nx.DiGraph()
        
        # 节点颜色映射
        node_colors = {}
        for node_id, node_info in graph["nodes"].items():
            # 查询节点使用红色，已知值节点使用蓝色，其他节点使用绿色
            if node_id == graph["query_node"]:
                node_colors[node_id] = "red"
            elif node_info["value"] is not None and node_info["value"] != "no_info":
                node_colors[node_id] = "blue"
            else:
                node_colors[node_id] = "green"
            
            # 添加节点和标签
            value_text = node_info["value"] if node_info["value"] is not None else "?"
            value_text = "" if value_text == "no_info" else f": {value_text}"
            node_label = f"{node_info['type']}{value_text}"
            G.add_node(node_id, label=node_label, color=node_colors[node_id])
        
        # 添加边
        for edge in graph["edges"]:
            G.add_edge(edge["from"], edge["to"], label=edge["relation"])
        
        # 创建图布局
        pos = nx.spring_layout(G, seed=42)  # 使用固定种子保持一致性
        
        # 创建图表
        plt.figure(figsize=(10, 8))
        
        # 绘制节点
        for node, color in node_colors.items():
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[node],
                node_color=color,
                node_size=2000,
                alpha=0.8
            )
        
        # 绘制边
        nx.draw_networkx_edges(
            G, pos,
            width=2,
            alpha=0.7,
            edge_color="gray",
            arrowsize=20
        )
        
        # 添加节点标签
        node_labels = {node: G.nodes[node]["label"] for node in G.nodes}
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=12,
            font_weight="bold"
        )
        
        # 添加边标签
        edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=10
        )
        
        # 设置图表
        plt.axis("off")
        plt.title("查询图结构")
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"查询图已保存至: {output_path}")
        return output_path
    
    def process_query(self, query):
        """处理自然语言查询并返回答案"""
        # 1. 解析查询
        query_graph = self.predict_query_graph(query)
        
        # 2. 从查询图中提取实体和关系
        main_entity = None
        for node_id, node in query_graph["nodes"].items():
            if node_id == "n0":  # 主实体通常是n0
                main_entity = node["value"]
                break
        
        if not main_entity or main_entity not in self.db:
            return f"抱歉，我没有关于{main_entity}的信息。"
        
        # 3. 提取查询关系
        relation = None
        query_node = query_graph["query_node"]
        for edge in query_graph["edges"]:
            if edge["to"] == query_node:
                relation = edge["relation"]
                break
        
        if not relation:
            return f"抱歉，我不理解您想查询{main_entity}的什么信息。"
            
        # 检查是否是设施查询
        if relation == "has_facility":
            return self.process_facility_query(query_graph, main_entity)
        
        # 4. 查询知识库
        university_data = self.db.get(main_entity)
        
        # 5. 根据关系获取对应的字段
        field = self.field_mapping.get(relation)
        
        if not field:
            return f"抱歉，我不了解{main_entity}的{relation}信息。"
        
        # 6. 处理字段数据获取
        value = self.get_field_value(university_data, field)
        if value is None:
            return f"抱歉，我没有关于{main_entity}的{relation}信息。"
            
        # 7. 使用generate_response生成格式化回答
        return self.generate_response(query_graph, value)
        
    def process_facility_query(self, query_graph, main_entity):
        """处理设施相关的查询"""
        # 获取查询图结构
        # 确定设施类型
        facility_type = None
        facility_node_id = None
        
        for node_id, node in query_graph["nodes"].items():
            if node_id != "n0" and node_id != query_graph["query_node"]:
                facility_type = node["type"]
                facility_node_id = node_id
                break
        
        if not facility_type:
            # 如果没有具体设施类型，返回所有设施信息
            if "facilities" in self.db[main_entity]:
                facilities = []
                for type_name, facility_info in self.db[main_entity]["facilities"].items():
                    facilities.append(f"{facility_info['name']}（位于{facility_info['location']}）")
                
                return f"{main_entity}的主要设施包括：{', '.join(facilities)}。"
            else:
                return f"抱歉，我没有关于{main_entity}设施的信息。"
        
        # 有具体设施类型的情况
        if "facilities" in self.db[main_entity] and facility_type in self.db[main_entity]["facilities"]:
            facility_info = self.db[main_entity]["facilities"][facility_type]
            
            # 确定查询的是设施的什么信息（位置、描述等）
            query_node = query_graph["nodes"][query_graph["query_node"]]
            query_type = query_node["type"]
            
            if query_type == "location":
                return f"{main_entity}的{facility_info['name']}位于{facility_info['location']}。"
            elif "description" in facility_info:
                return f"{main_entity}的{facility_info['name']}：{facility_info['description']}。"
            else:
                return f"{main_entity}有{facility_info['name']}，位于{facility_info['location']}。"
        
        return f"抱歉，我没有关于{main_entity}的{facility_type}设施信息。"
    
    def get_field_value(self, data, field):
        """灵活获取数据中的字段值，支持嵌套结构和默认值处理"""
        # 处理嵌套字段 (使用点号分隔)
        if "." in field:
            parts = field.split(".", 1)
            if parts[0] in data and isinstance(data[parts[0]], dict):
                return self.get_field_value(data[parts[0]], parts[1])
            return self.get_default_value(field)
        
        # 处理facilities特殊字段
        if field == "has_facility" and "facilities" in data:
            facilities_list = []
            for facility_type, facility_info in data["facilities"].items():
                facilities_list.append(f"{facility_info['name']}（位于{facility_info['location']}）")
            return "、".join(facilities_list)
            
        # 处理字典类型字段
        if field in data:
            if isinstance(data[field], dict):
                # 处理facilities字段
                if field == "facilities":
                    facilities_list = []
                    for facility_type, facility_info in data[field].items():
                        facilities_list.append(f"{facility_info['name']}（位于{facility_info['location']}）")
                    return "、".join(facilities_list)
                # 处理location特殊字段
                elif "address" in data[field]:
                    return data[field]["address"]
                elif "city" in data[field]:
                    return data[field]["city"]
                else:
                    # 将字典转换为字符串
                    return ", ".join([f"{k}: {v}" for k, v in data[field].items()])
            return data[field]
        
        # 处理缺失字段
        return self.get_default_value(field)
        
    def get_default_value(self, field):
        """获取字段的默认值，如果配置中存在"""
        if "default_values" in self.config and field in self.config["default_values"]:
            return self.config["default_values"][field]
        return "暂无相关信息"


def evaluate(model, val_loader, device, relation_criterion=None, query_criterion=None):
    """评估模型性能"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    entity_correct = 0
    relation_correct = 0
    query_type_correct = 0
    total = 0
    
    # 如果没有提供特定的损失函数，使用交叉熵
    if relation_criterion is None:
        relation_criterion = criterion
    if query_criterion is None:
        query_criterion = criterion
    
    with torch.no_grad():
        for batch in val_loader:
            # 获取数据
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            main_entity_type = batch["main_entity_type"].to(device)
            relation = batch["relation"].to(device)
            query_type = batch["query_type"].to(device)
            
            # 前向传播
            main_entity_logits, relation_logits, query_type_logits = model(input_ids, attention_mask)
            
            # 计算损失
            entity_loss = criterion(main_entity_logits, main_entity_type)
            relation_loss = relation_criterion(relation_logits, relation)
            query_type_loss = query_criterion(query_type_logits, query_type)
            
            # 总损失 - 与训练阶段使用相同的权重
            loss = entity_loss + 1.5 * relation_loss + 1.2 * query_type_loss
            
            # 累加批次损失
            val_loss += loss.item()
            
            # 统计准确率
            _, entity_pred = torch.max(main_entity_logits, dim=1)
            _, relation_pred = torch.max(relation_logits, dim=1)
            _, query_pred = torch.max(query_type_logits, dim=1)
            
            entity_correct += (entity_pred == main_entity_type).sum().item()
            relation_correct += (relation_pred == relation).sum().item()
            query_type_correct += (query_pred == query_type).sum().item()
            total += main_entity_type.size(0)
    
    # 计算平均损失和准确率
    val_loss /= len(val_loader)
    val_entity_acc = entity_correct / total
    val_relation_acc = relation_correct / total
    val_query_type_acc = query_type_correct / total
    
    return val_loss, val_entity_acc, val_relation_acc, val_query_type_acc


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, patience=5, model_save_path="model/graph_parser/output/best_graph_parser.pth"):
    # 增加早停耐心值
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}
    
    # 获取节点类型和关系的总数
    num_node_types = model.main_entity_decoder[-1].out_features
    num_relations = model.relation_decoder[-1].out_features
    
    # 计算关系类型和查询类型的类别权重
    relation_weights = compute_class_weights(train_loader, "relation", num_relations)
    query_type_weights = compute_class_weights(train_loader, "query_type", num_node_types)
    
    # 使用带权重的损失函数，并确保权重在正确的设备上
    relation_criterion = nn.CrossEntropyLoss(weight=relation_weights.to(device))
    query_type_criterion = nn.CrossEntropyLoss(weight=query_type_weights.to(device))
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        entity_correct = 0
        relation_correct = 0
        query_type_correct = 0
        total = 0
        
        # 记录每个批次的进度
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in progress_bar:
            # 获取数据
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            main_entity_type = batch["main_entity_type"].to(device)
            relation = batch["relation"].to(device)
            query_type = batch["query_type"].to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            main_entity_logits, relation_logits, query_type_logits = model(input_ids, attention_mask)
            
            # 计算损失 - 带有不同权重
            entity_loss = criterion(main_entity_logits, main_entity_type)
            relation_loss = relation_criterion(relation_logits, relation)
            query_type_loss = query_type_criterion(query_type_logits, query_type)
            
            # 总损失 - 加大关系和查询类型损失的权重
            loss = entity_loss + 1.5 * relation_loss + 1.2 * query_type_loss
            
            # 添加L2正则化项，增强对关系和实体类型的兼容性学习
            compatibility_reg = 0.01 * torch.norm(model.compatibility_matrix, p=2)
            loss += compatibility_reg
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            # 统计准确率
            _, entity_pred = torch.max(main_entity_logits, dim=1)
            _, relation_pred = torch.max(relation_logits, dim=1)
            _, query_pred = torch.max(query_type_logits, dim=1)
            
            entity_correct += (entity_pred == main_entity_type).sum().item()
            relation_correct += (relation_pred == relation).sum().item()
            query_type_correct += (query_pred == query_type).sum().item()
            total += main_entity_type.size(0)
            
            # 累加批次损失
            train_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "entity_acc": f"{entity_correct/total:.4f}",
                "relation_acc": f"{relation_correct/total:.4f}",
                "query_acc": f"{query_type_correct/total:.4f}"
            })
        
        # 计算训练集平均损失和准确率
        train_loss /= len(train_loader)
        train_entity_acc = entity_correct / total
        train_relation_acc = relation_correct / total
        train_query_type_acc = query_type_correct / total
        
        # 验证阶段
        val_loss, val_entity_acc, val_relation_acc, val_query_type_acc = evaluate(model, val_loader, device, 
                                                                     relation_criterion, query_type_criterion)
        
        # 更新学习率
        scheduler.step()
        
        # 保存历史记录
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        # 打印结果
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Entity Acc: {train_entity_acc:.4f}, Relation Acc: {train_relation_acc:.4f}, Query Acc: {train_query_type_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Entity Acc: {val_entity_acc:.4f}, Relation Acc: {val_relation_acc:.4f}, Query Acc: {val_query_type_acc:.4f}")
        
        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            # 保存最佳模型
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停! 在第{best_epoch}轮达到最佳验证损失.")
                break
    
    # 返回训练历史
    return history


def compute_class_weights(dataloader, field, num_classes=None):
    """计算类别权重，用于处理数据不平衡问题"""
    class_counts = defaultdict(int)
    
    # 统计每个类别的样本数
    for batch in dataloader:
        labels = batch[field].numpy()
        for label in labels:
            class_counts[label] += 1
    
    # 计算类别权重 (反比于样本数量)
    total_samples = sum(class_counts.values())
    
    # 确保num_classes至少等于最大类ID+1
    if num_classes is None:
        num_classes = max(class_counts.keys()) + 1
    else:
        num_classes = max(num_classes, max(class_counts.keys()) + 1)
    
    # 初始化权重数组，确保大小匹配类总数
    weights = np.ones(num_classes)
    
    # 为出现的类设置权重
    for class_idx, count in class_counts.items():
        if count > 0:
            weights[class_idx] = total_samples / (count * len(class_counts))
    
    # 确保返回torch.float32类型的权重
    return torch.tensor(weights, dtype=torch.float32)


def detailed_evaluation(model, test_loader, id2type, id2relation, device):
    """详细评估模型性能，按类型和关系分析"""
    model.eval()
    
    # 准确率统计
    entity_correct = 0
    relation_correct = 0
    query_type_correct = 0
    total = 0
    
    # 混淆矩阵
    entity_confusion = {}  # {真实值: {预测值: 计数}}
    relation_confusion = {}
    query_type_confusion = {}
    
    # 例子记录
    examples = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            main_entity_type = batch["main_entity_type"].to(device)
            relation = batch["relation"].to(device)
            query_type = batch["query_type"].to(device)
            questions = batch["question"]
            
            main_entity_logits, relation_logits, query_type_logits = model(input_ids, attention_mask)
            
            # 获取预测
            entity_pred = torch.argmax(main_entity_logits, dim=1)
            relation_pred = torch.argmax(relation_logits, dim=1)
            query_pred = torch.argmax(query_type_logits, dim=1)
            
            # 添加额外的逻辑规则约束 - 检查关系和查询类型的兼容性
            for i in range(len(relation_pred)):
                rel_id = relation_pred[i].item()
                
                # 获取与此关系兼容的查询类型
                compatible_query_types = get_compatible_query_types(rel_id, id2relation)
                
                # 如果当前预测的查询类型与关系不兼容，选择最兼容的查询类型
                pred_query_id = query_pred[i].item()
                if id2type.get(pred_query_id) not in compatible_query_types:
                    # 重新选择最兼容的查询类型
                    for j, logit in enumerate(query_type_logits[i]):
                        if id2type.get(j) in compatible_query_types:
                            query_pred[i] = j
                            break
            
            # 更新准确率
            entity_correct += (entity_pred == main_entity_type).sum().item()
            relation_correct += (relation_pred == relation).sum().item()
            query_type_correct += (query_pred == query_type).sum().item()
            
            # 更新混淆矩阵
            for true_e, pred_e, true_r, pred_r, true_q, pred_q, question in zip(
                main_entity_type.cpu().numpy(), 
                entity_pred.cpu().numpy(),
                relation.cpu().numpy(),
                relation_pred.cpu().numpy(),
                query_type.cpu().numpy(),
                query_pred.cpu().numpy(),
                questions
            ):
                # 实体类型混淆
                true_e_str = id2type.get(int(true_e), "<unk>")
                pred_e_str = id2type.get(int(pred_e), "<unk>")
                if true_e_str not in entity_confusion:
                    entity_confusion[true_e_str] = {}
                if pred_e_str not in entity_confusion[true_e_str]:
                    entity_confusion[true_e_str][pred_e_str] = 0
                entity_confusion[true_e_str][pred_e_str] += 1
                
                # 关系混淆
                true_r_str = id2relation.get(int(true_r), "<unk>")
                pred_r_str = id2relation.get(int(pred_r), "<unk>")
                if true_r_str not in relation_confusion:
                    relation_confusion[true_r_str] = {}
                if pred_r_str not in relation_confusion[true_r_str]:
                    relation_confusion[true_r_str][pred_r_str] = 0
                relation_confusion[true_r_str][pred_r_str] += 1
                
                # 查询类型混淆
                true_q_str = id2type.get(int(true_q), "<unk>")
                pred_q_str = id2type.get(int(pred_q), "<unk>")
                if true_q_str not in query_type_confusion:
                    query_type_confusion[true_q_str] = {}
                if pred_q_str not in query_type_confusion[true_q_str]:
                    query_type_confusion[true_q_str][pred_q_str] = 0
                query_type_confusion[true_q_str][pred_q_str] += 1
                
                # 记录错误例子
                if true_e != pred_e or true_r != pred_r or true_q != pred_q:
                    examples.append({
                        "question": question,
                        "true_entity": true_e_str,
                        "pred_entity": pred_e_str,
                        "true_relation": true_r_str,
                        "pred_relation": pred_r_str,
                        "true_query": true_q_str,
                        "pred_query": pred_q_str
                    })
            
            total += len(main_entity_type)
    
    # 打印总体准确率
    print(f"\n=== 总体准确率 ===")
    print(f"实体类型准确率: {entity_correct/total:.4f}")
    print(f"关系准确率: {relation_correct/total:.4f}")
    print(f"查询类型准确率: {query_type_correct/total:.4f}")
    
    # 打印混淆最严重的类别
    print("\n=== 混淆最严重的实体类型 ===")
    for true_type, preds in entity_confusion.items():
        incorrect = sum(count for pred, count in preds.items() if pred != true_type)
        total_type = sum(preds.values())
        if total_type > 0 and incorrect/total_type > 0.3:  # 错误率超过30%
            print(f"{true_type}:")
            for pred, count in sorted(preds.items(), key=lambda x: x[1], reverse=True):
                if pred != true_type:
                    print(f"  误判为 {pred}: {count}次 ({count/total_type:.2%})")
    
    # 打印混淆最严重的关系
    print("\n=== 混淆最严重的关系 ===")
    for true_rel, preds in relation_confusion.items():
        incorrect = sum(count for pred, count in preds.items() if pred != true_rel)
        total_rel = sum(preds.values())
        if total_rel > 0 and incorrect/total_rel > 0.3:  # 错误率超过30%
            print(f"{true_rel}:")
            for pred, count in sorted(preds.items(), key=lambda x: x[1], reverse=True):
                if pred != true_rel:
                    print(f"  误判为 {pred}: {count}次 ({count/total_rel:.2%})")
    
    # 打印几个错误例子
    print("\n=== 典型错误例子 ===")
    for i, example in enumerate(examples[:5]):  # 只显示前5个
        print(f"例{i+1}: {example['question']}")
        print(f"  实体类型: {example['true_entity']} → {example['pred_entity']}")
        print(f"  关系: {example['true_relation']} → {example['pred_relation']}")
        print(f"  查询类型: {example['true_query']} → {example['pred_query']}")
    
    return entity_correct/total, relation_correct/total, query_type_correct/total


def get_compatible_query_types(relation_id, id2relation):
    """获取与关系兼容的查询类型"""
    relation_str = id2relation.get(relation_id, "<unk>")
    
    # 关系到查询类型的映射字典
    relation_to_query_types = {
        "has_phone": ["phone"],
        "located_in": ["location", "city"],
        "has_president": ["person"],
        "has_website": ["website"],
        "has_tuition": ["tuition"],
        "has_motto": ["motto"],
        "founded_in": ["year"],
        "has_student_count": ["student_count", "number"],
        "has_admission_score": ["admission_score", "number"],
        "has_international_program": ["international_program"],
        "predecessor": ["university"],
        "nearest_metro": ["metro_station"],
        "anniversary_date": ["date"],
        "academician_count": ["number"],
        "library_volume": ["number"],
        "party_secretary": ["person"],
        "offers_major": ["major_list", "major"],
        "has_facility": ["library", "sports_center", "canteen", "laboratory", "dormitory"],
        "located_at": ["location"],
        "has_quality": ["quality"]
    }
    
    return relation_to_query_types.get(relation_str, ["<unk>"])


# 示例使用
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    print("加载训练数据...")
    # 1. 准备训练数据
    with open("data/graph_parser/training_data.json", "r", encoding="utf-8") as f:
        training_data = json.load(f)
    
    # 2. 划分数据集
    train_examples, test_examples = train_test_split(training_data, test_size=0.2, random_state=42)
    train_examples, val_examples = train_test_split(train_examples, test_size=0.1, random_state=42)
    
    print(f"数据集大小: 训练集 {len(train_examples)}, 验证集 {len(val_examples)}, 测试集 {len(test_examples)}")
    
    # 3. 初始化tokenizer和模型
    print("初始化模型...")
    pretrained_model = "hfl/chinese-roberta-wwm-ext"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    # 4. 创建数据集
    train_dataset = GraphQueryDataset(train_examples, tokenizer)
    val_dataset = GraphQueryDataset(val_examples, tokenizer)
    test_dataset = GraphQueryDataset(test_examples, tokenizer)
    
    # 5. 获取节点类型和关系的词汇表
    node_types = train_dataset.node_types
    relations = train_dataset.relations
    
    print(f"节点类型数量: {len(node_types)}")
    print(f"关系类型数量: {len(relations)}")
    
    # 6. 创建数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 7. 初始化模型
    device = torch.device("musa" if torch_musa.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = GraphQueryParser(
        pretrained_model=pretrained_model,
        hidden_dim=768,
        num_node_types=len(node_types),
        num_relations=len(relations),
        dropout_rate=0.2
    )
    model.to(device)
    
    # 8. 训练模型
    print("开始训练模型...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # 决定是否训练模型（如果已有预训练模型，可以跳过）
    train_new_model = True
    if train_new_model:
        history = train_model(
            model, 
            train_loader, 
            val_loader, 
            optimizer, 
            scheduler, 
            num_epochs=20, 
            device=device, 
            patience=5
        )
    else:
        # 加载预训练模型
        try:
            model.load_state_dict(torch.load("model/graph_parser/output/best_graph_parser.pth", map_location=device))
            print("已加载预训练模型")
        except:
            print("无法加载预训练模型，使用初始化模型")
    
    # 9. 评估模型
    print("\n开始详细评估...")
    id2type = {v: k for k, v in node_types.items()}
    id2relation = {v: k for k, v in relations.items()}
    
    entity_acc, relation_acc, query_acc = detailed_evaluation(
        model, test_loader, id2type, id2relation, device
    )
    
    print(f"\n总体测试集性能:")
    
    # 9. 创建查询系统
    query_system = GraphQuerySystem(model, tokenizer, "config.json")
    
    # 10. 处理查询
    query = "清华大学的电话号码是多少？"
    response = query_system.process_query(query)
    print(f"问题: {query}")
    print(f"回答: {response}")