import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from model import GraphQueryParser, GraphQueryDataset, GraphQuerySystem, detailed_evaluation

def load_test_data():
    """加载测试数据和词汇映射"""
    print("加载测试数据...")
    try:
        with open("data/graph_parser/test_examples.json", "r", encoding="utf-8") as f:
            test_examples = json.load(f)
    except FileNotFoundError:
        # 如果找不到测试集文件，从原始数据中分割
        with open("data/graph_parser/training_data.json", "r", encoding="utf-8") as f:
            training_data = json.load(f)
        from sklearn.model_selection import train_test_split
        _, test_examples = train_test_split(training_data, test_size=0.2, random_state=42)
    
    print("加载词汇映射...")
    try:
        with open("data/graph_parser/node_types.json", "r", encoding="utf-8") as f:
            node_types = json.load(f)
        with open("data/graph_parser/relations.json", "r", encoding="utf-8") as f:
            relations = json.load(f)
    except FileNotFoundError:
        print("找不到词汇映射文件，将从测试数据中创建")
        pretrained_model = "hfl/chinese-roberta-wwm-ext"
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        temp_dataset = GraphQueryDataset(test_examples, tokenizer)
        node_types = temp_dataset.node_types
        relations = temp_dataset.relations
        
    return test_examples, node_types, relations

def init_model_and_tokenizer(node_types, relations, device):
    """初始化并加载模型"""
    print("初始化模型...")
    pretrained_model = "hfl/chinese-roberta-wwm-ext"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    model = GraphQueryParser(
        pretrained_model=pretrained_model,
        hidden_dim=768,
        num_node_types=len(node_types),
        num_relations=len(relations),
        dropout_rate=0.2
    )
    model.to(device)
    
    try:
        model.load_state_dict(torch.load("best_graph_parser.pth", map_location=device))
        print("已加载预训练模型")
    except:
        print("警告：无法加载预训练模型，使用初始化模型")
        
    return model, tokenizer

def evaluate_model(model, test_loader, node_types, relations, device):
    """评估模型性能"""
    print("\n开始详细评估...")
    id2type = {v: k for k, v in node_types.items()}
    id2relation = {v: k for k, v in relations.items()}
    
    entity_acc, relation_acc, query_acc = detailed_evaluation(
        model, test_loader, id2type, id2relation, device
    )
    
    print(f"\n总体测试集性能:")
    print(f"实体类型准确率: {entity_acc:.4f}")
    print(f"关系准确率: {relation_acc:.4f}")
    print(f"查询类型准确率: {query_acc:.4f}")

def test_queries(query_system):
    """测试样例查询"""
    print("\n测试样例查询...")
    test_queries = [
        "清华大学的电话号码是多少？",
        "北京大学在哪里？",
        "请问中国科学技术大学的校长是谁？",
        "复旦大学的网站是什么？",
        "武汉大学什么时候成立的？",
        "浙江大学的校训是什么？",
        "上海交通大学的学生人数是多少？",
        "南京大学的占地面积有多大？",
        "哈尔滨工业大学的简称是什么？",
        "西安交通大学的前身是什么学校？",
        "同济大学的地铁站是哪一站？",
        "天津大学的校庆日是哪一天？",
        "东南大学的院士人数是多少？",
        "四川大学的图书馆藏书量是多少？",
        "中山大学的现任党委书记是谁？"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n===== 测试查询 {i+1} =====")
        print(f"问题: {query}")
        response = query_system.process_query(query)
        print(f"回答: {response}")
        
        output_path = f"outputs/query_{i+1}.png"
        query_graph = query_system.predict_query_graph(query)
        query_system.visualize_graph_image(query_graph, output_path=output_path)
        print(f"查询图已保存至: {output_path}")

def init_query_system(model, tokenizer, node_types, relations, device):
    """初始化查询系统"""
    query_system = GraphQuerySystem(model, tokenizer, node_types, relations, device)
    
    # # 确保加载知识库
    # if not hasattr(query_system, 'knowledge_base') or not query_system.knowledge_base:
    #     query_system.load_knowledge_base()
    
    return query_system

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    
    # 1. 加载数据
    test_examples, node_types, relations = load_test_data()
    
    # 2. 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 3. 初始化模型和tokenizer
    model, tokenizer = init_model_and_tokenizer(node_types, relations, device)
    
    # 4. 创建测试数据集和加载器
    test_dataset = GraphQueryDataset(test_examples, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # 5. 评估模型
    evaluate_model(model, test_loader, node_types, relations, device)
    
    # 6. 初始化查询系统并加载知识库
    query_system = init_query_system(model, tokenizer, node_types, relations, device)
    
    # 7. 测试查询
    test_queries(query_system)
    
    print("\n测试完成！")