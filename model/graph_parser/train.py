import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shutil

from model import GraphQueryParser, GraphQueryDataset, train_model

def train_graph_parser():
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("data/graph_parser", exist_ok=True)
    
    print("加载训练数据...")
    # 1. 准备训练数据
    try:
        # 首选加载增强数据
        with open("data/graph_parser/training_data.json", "r", encoding="utf-8") as f:
            training_data = json.load(f)
            print(f"使用增强训练数据: {len(training_data)}个样例")
    except FileNotFoundError:
        # 退回到原始数据
        with open("data/graph_parser/training_data.json", "r", encoding="utf-8") as f:
            training_data = json.load(f)
            print(f"使用原始训练数据: {len(training_data)}个样例")
    
    # 2. 划分数据集
    train_examples, test_examples = train_test_split(training_data, test_size=0.15, random_state=42)
    train_examples, val_examples = train_test_split(train_examples, test_size=0.1, random_state=42)
    
    print(f"数据集大小: 训练集 {len(train_examples)}, 验证集 {len(val_examples)}, 测试集 {len(test_examples)}")
    
    # 3. 初始化tokenizer和模型
    print("初始化模型...")
    pretrained_model = "hfl/chinese-roberta-wwm-ext"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    # 4. 创建数据集
    train_dataset = GraphQueryDataset(train_examples, tokenizer)
    val_dataset = GraphQueryDataset(val_examples, tokenizer)
    
    # 保存测试集供以后使用
    with open("data/graph_parser/test_examples.json", "w", encoding="utf-8") as f:
        json.dump(test_examples, f, ensure_ascii=False, indent=2)
    
    # 保存映射表供测试使用
    with open("data/graph_parser/node_types.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset.node_types, f, ensure_ascii=False, indent=2)
    
    with open("data/graph_parser/relations.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset.relations, f, ensure_ascii=False, indent=2)
    
    # 同时在当前目录复制一份(解决找不到问题)
    shutil.copy("data/graph_parser/node_types.json", "./node_types.json")
    shutil.copy("data/graph_parser/relations.json", "./relations.json")
    
    # 5. 获取节点类型和关系的词汇表
    node_types = train_dataset.node_types
    relations = train_dataset.relations
    
    print(f"节点类型数量: {len(node_types)}")
    print(f"关系类型数量: {len(relations)}")
    print(f"节点类型: {list(node_types.keys())}")
    print(f"关系类型: {list(relations.keys())}")
    
    # 6. 创建数据加载器
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 7. 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = GraphQueryParser(
        pretrained_model=pretrained_model,
        hidden_dim=768,
        num_node_types=len(node_types),
        num_relations=len(relations),
        dropout_rate=0.3  # 增加dropout以减少过拟合
    )
    model.to(device)
    
    # 8. 训练模型
    print("开始训练模型...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        num_epochs=10,  # 增加轮数
        device=device, 
        patience=7  # 增加早停耐心值
    )
    
    # 9. 保存训练历史
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
    plt.savefig("outputs/training_history.png")
    plt.close()
    
    print("训练完成！模型已保存为 best_graph_parser.pth")
    
    return model, tokenizer, node_types, relations

if __name__ == "__main__":
    train_graph_parser()
