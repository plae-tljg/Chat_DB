import torch
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
        
        # 添加注意力层
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True, dropout=dropout_rate)
        
        # 主实体预测器
        self.main_entity_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_dim//2, num_node_types)
        )
        
        # 修改: 关系预测器接收主实体表示作为附加输入
        self.relation_decoder = nn.Sequential(
            nn.Linear(hidden_dim + num_node_types, hidden_dim//2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_dim//2, num_relations)
        )
        
        # 修改: 查询类型预测器接收主实体和关系表示作为附加输入
        self.query_type_decoder = nn.Sequential(
            nn.Linear(hidden_dim + num_node_types + num_relations, hidden_dim//2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(hidden_dim//2, num_node_types)
        )
        
    def forward(self, input_ids, attention_mask):
        # 编码文本
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取每个token的表示
        token_repr = outputs.last_hidden_state
        
        # 应用自注意力
        attn_output, _ = self.attention(token_repr, token_repr, token_repr, 
                                      key_padding_mask=(attention_mask == 0))
        
        # 残差连接
        enhanced_repr = token_repr + attn_output
        
        # 使用[CLS]令牌表示
        sentence_repr = enhanced_repr[:, 0, :]
        
        # 转换表示
        transformed = self.transform(sentence_repr)
        
        # 修改: 使用级联预测，每个预测任务都依赖前一个任务的结果
        
        # 1. 预测主实体类型
        main_entity_logits = self.main_entity_decoder(transformed)
        main_entity_probs = F.softmax(main_entity_logits, dim=1)
        
        # 2. 预测关系类型，加入主实体信息
        # 将主实体概率分布与句子表示连接起来
        relation_input = torch.cat([transformed, main_entity_probs], dim=1)
        relation_logits = self.relation_decoder(relation_input)
        relation_probs = F.softmax(relation_logits, dim=1)
        
        # 3. 预测查询类型，同时考虑主实体和关系信息
        # 将主实体概率分布、关系概率分布与句子表示连接起来
        query_type_input = torch.cat([transformed, main_entity_probs, relation_probs], dim=1)
        query_type_logits = self.query_type_decoder(query_type_input)
        
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
    def __init__(self, model, tokenizer, node_types, relations, device):
        self.model = model
        self.tokenizer = tokenizer
        self.node_types = node_types
        self.relations = relations
        self.device = device
        
        # 创建ID到类型/关系的映射
        self.id2type = {v: k for k, v in node_types.items()}
        self.id2relation = {v: k for k, v in relations.items()}
        
        # 加载知识库
        self.db = self.load_knowledge_base()
        
        # 响应模板
        self.templates = {
            "phone": "{entity}的电话号码是{value}。",
            "location": "{entity}位于{value}。",
            "website": "{entity}的官网是{value}。",
            "person": "{entity}的校长是{value}。",
            "tuition": "{entity}的学费是{value}。",
            "founded_in": "{entity}成立于{value}。",
            "motto": "{entity}的校训是{value}。",
            "student_count": "{entity}的学生人数是{value}人。",
            "area": "{entity}的占地面积是{value}。",
            "short_name": "{entity}的简称是{value}。",
            "predecessor": "{entity}的前身是{value}。",
            "nearest_metro": "{entity}附近的地铁站是{value}。",
            "anniversary_date": "{entity}的校庆日是{value}。",
            "academician_count": "{entity}的院士人数是{value}人。",
            "library_volume": "{entity}的图书馆藏书量是{value}。",
            "party_secretary": "{entity}的党委书记是{value}。"
        }
        
        # 关系到数据库字段的映射
        self.field_mapping = {
            "has_phone": "phone",
            "located_in": "location",
            "has_website": "website",
            "has_president": "president",
            "has_tuition": "tuition",
            "founded_in": "founding_date",
            "has_motto": "motto",
            "has_student_count": "student_count",
            "has_area": "area",
            "has_short_name": "short_name",
            "has_predecessor": "predecessor",
            "has_nearest_metro": "nearest_metro",
            "has_anniversary_date": "anniversary_date",
            "has_academician_count": "academician_count",
            "has_library_volume": "library_volume",
            "has_party_secretary": "party_secretary"
        }
        
        # 查询类型到响应模板的映射
        self.type_mapping = {
            "phone": "phone",
            "location": "location",
            "website": "website",
            "person": "president",
            "tuition": "tuition",
            "year": "founding_date",
            "motto": "motto",
            "student_count": "student_count",
            "area": "area",
            "short_name": "short_name",
            "predecessor": "predecessor",
            "nearest_metro": "nearest_metro",
            "date": "anniversary_date",
            "academician_count": "academician_count",
            "library": "library_volume",
            "party_secretary": "party_secretary"
        }
    
    def load_knowledge_base(self):
        """从JSON文件加载大学知识库"""
        try:
            knowledge_base_path = "data/graph_parser/university_knowledge_base.json"
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
        """使用模型解析文本为查询图结构"""
        return self.model.predict_query_graph(text, self.tokenizer, self.node_types, self.relations, self.device)
    
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
        for edge in query_graph["edges"]:
            if edge["to"] == query_graph["query_node"]:
                template_key = self.type_mapping.get(query_type)
                if template_key and template_key in self.templates:
                    return self.templates[template_key].format(entity=main_value, value=result)
        
        # 默认回复
        return f"{main_value}的{query_type}是{result}。"
    
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
        
        # 4. 查询知识库
        university_data = self.db.get(main_entity)
        
        # 5. 根据关系获取对应的字段
        field = self.field_mapping.get(relation)
        
        if not field:
            return f"抱歉，我不了解{main_entity}的{relation}信息。"
        
        # 6. 处理特殊字段
        if field == "location":
            # 对于location字段，需要获取address子字段
            location_data = university_data.get("location", {})
            value = location_data.get("address", "未知地址")
            return f"{main_entity}位于{value}。"
        else:
            # 对于其他字段，直接获取字段值
            value = university_data.get(field, "未知")
            
            # 查找关系对应的模板
            template_key = self.type_mapping.get(field, field)
            if template_key in self.templates:
                return self.templates[template_key].format(entity=main_entity, value=value)
            else:
                return f"{main_entity}的{field}是{value}。"


def evaluate(model, val_loader, device):
    """评估模型在验证集上的性能"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            main_entity_type = batch["main_entity_type"].to(device)
            relation = batch["relation"].to(device)
            query_type = batch["query_type"].to(device)
            
            main_entity_logits, relation_logits, query_type_logits = model(input_ids, attention_mask)
            
            # 计算损失
            loss_entity = F.cross_entropy(main_entity_logits, main_entity_type)
            loss_relation = F.cross_entropy(relation_logits, relation)
            loss_query = F.cross_entropy(query_type_logits, query_type)
            
            loss = loss_entity + loss_relation + loss_query
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


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
        total = sum(preds.values())
        if total > 0 and incorrect/total > 0.3:  # 错误率超过30%
            print(f"{true_type}:")
            for pred, count in sorted(preds.items(), key=lambda x: x[1], reverse=True):
                if pred != true_type:
                    print(f"  误判为 {pred}: {count}次 ({count/total:.2%})")
    
    # 打印混淆最严重的关系
    print("\n=== 混淆最严重的关系 ===")
    for true_rel, preds in relation_confusion.items():
        incorrect = sum(count for pred, count in preds.items() if pred != true_rel)
        total = sum(preds.values())
        if total > 0 and incorrect/total > 0.3:  # 错误率超过30%
            print(f"{true_rel}:")
            for pred, count in sorted(preds.items(), key=lambda x: x[1], reverse=True):
                if pred != true_rel:
                    print(f"  误判为 {pred}: {count}次 ({count/total:.2%})")
    
    # 打印几个错误例子
    print("\n=== 典型错误例子 ===")
    for i, example in enumerate(examples[:5]):  # 只显示前5个
        print(f"例{i+1}: {example['question']}")
        print(f"  实体类型: {example['true_entity']} → {example['pred_entity']}")
        print(f"  关系: {example['true_relation']} → {example['pred_relation']}")
        print(f"  查询类型: {example['true_query']} → {example['pred_query']}")
    
    return entity_correct/total, relation_correct/total, query_type_correct/total


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device, patience=5):
    # 增加早停耐心值
    best_val_loss = float('inf')
    no_improve = 0
    
    # 记录训练历史
    history = {
        "train_loss": [],
        "val_loss": []
    }
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            main_entity_type = batch["main_entity_type"].to(device)
            relation = batch["relation"].to(device)
            query_type = batch["query_type"].to(device)
            
            # 常规训练流程
            main_entity_logits, relation_logits, query_type_logits = model(input_ids, attention_mask)
            
            # 添加标签平滑
            loss_entity = F.cross_entropy(main_entity_logits, main_entity_type, label_smoothing=0.1)
            loss_relation = F.cross_entropy(relation_logits, relation, label_smoothing=0.1)
            loss_query = F.cross_entropy(query_type_logits, query_type, label_smoothing=0.1)
            
            loss = loss_entity + loss_relation + loss_query
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, device)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        # 更新学习率
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
        
        # 检查早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_graph_parser.pth")
            print("保存最佳模型")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"早停: {patience}轮无改善")
                break
    
    # 绘制训练曲线
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei'] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="训练损失")
    plt.plot(history["val_loss"], label="验证损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("训练与验证损失")
    plt.savefig("training_history.png")
    plt.close()
    
    print(f"训练完成，最佳验证损失: {best_val_loss:.4f}")
    return history


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            model.load_state_dict(torch.load("best_graph_parser.pth", map_location=device))
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
    query_system = GraphQuerySystem(model, tokenizer, node_types, relations, device)
    
    # 10. 处理查询
    query = "清华大学的电话号码是多少？"
    response = query_system.process_query(query)
    print(f"问题: {query}")
    print(f"回答: {response}")