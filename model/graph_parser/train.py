import os
import json
import torch
import torch_musa
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shutil
import numpy as np
from torch.utils.data import WeightedRandomSampler

from model import GraphQueryParser, GraphQueryDataset, train_model

def train_graph_parser():
    # 获取项目根目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # 创建输出目录
    output_dir = os.path.join(project_root, "outputs")
    data_dir = os.path.join(project_root, "data/graph_parser")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    print("加载训练数据...")
    # 1. 准备训练数据
    training_data_path = os.path.join(data_dir, "training_data.json")
    with open(training_data_path, "r", encoding="utf-8") as f:
        training_data = json.load(f)
        print(f"使用原始训练数据: {len(training_data)}个样例")

    # 对数据进行分析，获取关系分布
    relation_counts = {}
    for item in training_data:
        relation = item["graph"]["edges"][0]["relation"]
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    
    print("\n关系分布情况:")
    for relation, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{relation}: {count}个样例")
    
    # 2. 平衡数据集 - 对低频关系类型进行数据增强
    print("\n执行数据平衡...")
    balanced_data = balance_training_data(training_data, relation_counts)
    print(f"平衡后的数据集大小: {len(balanced_data)}个样例")
    
    # 3. 划分数据集
    train_examples, test_examples = train_test_split(balanced_data, test_size=0.15, random_state=42)
    train_examples, val_examples = train_test_split(train_examples, test_size=0.1, random_state=42)
    
    print(f"数据集大小: 训练集 {len(train_examples)}, 验证集 {len(val_examples)}, 测试集 {len(test_examples)}")
    
    # 4. 初始化tokenizer和模型
    print("初始化模型...")
    pretrained_model = "hfl/chinese-roberta-wwm-ext"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    # 5. 创建数据集
    train_dataset = GraphQueryDataset(train_examples, tokenizer)
    val_dataset = GraphQueryDataset(val_examples, tokenizer)
    
    # 保存测试集供以后使用
    test_examples_path = os.path.join(data_dir, "train_output/test_examples.json")
    with open(test_examples_path, "w", encoding="utf-8") as f:
        json.dump(test_examples, f, ensure_ascii=False, indent=2)
    
    # 保存映射表供测试使用
    node_types_path = os.path.join(data_dir, "train_output/node_types.json")
    with open(node_types_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset.node_types, f, ensure_ascii=False, indent=2)
    
    relations_path = os.path.join(data_dir, "train_output/relations.json")
    with open(relations_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset.relations, f, ensure_ascii=False, indent=2)

    # 6. 获取节点类型和关系的词汇表
    node_types = train_dataset.node_types
    relations = train_dataset.relations
    
    print(f"节点类型数量: {len(node_types)}")
    print(f"关系类型数量: {len(relations)}")
    print(f"节点类型: {list(node_types.keys())}")
    print(f"关系类型: {list(relations.keys())}")
    
    # 7. 创建数据加载器
    batch_size = 32  # 增大批次大小
    
    # 使用加权采样器平衡训练数据
    samples_weights = get_sample_weights(train_examples)
    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 8. 初始化模型
    # 检查MUSA是否可用并打印详细信息
    if torch_musa.is_available():
        print(f"MUSA可用，版本: {torch_musa.__version__}")
        device_count = torch_musa.device_count()
        print(f"MUSA设备数量: {device_count}")
        for i in range(device_count):
            prop = torch_musa.get_device_properties(i)
            print(f"MUSA设备 #{i}: {prop}")
    else:
        print("MUSA不可用")
    
    device = torch.device("musa" if torch_musa.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = GraphQueryParser(
        pretrained_model=pretrained_model,
        hidden_dim=768,
        num_node_types=len(node_types),
        num_relations=len(relations),
        dropout_rate=0.3  # 增加dropout以减少过拟合
    )
    model.to(device)
    
    # 9. 训练模型
    print("开始训练模型...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # 使用余弦退火学习率调度
    total_steps = len(train_loader) * 15  # 15个epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # 保存模型到绝对路径
    model_save_path = os.path.join(project_root, "best_graph_parser.pth")
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        num_epochs=15,  # 增加轮数
        device=device, 
        patience=5,  # 早停耐心值
        model_save_path=model_save_path
    )
    
    # 10. 保存训练历史
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
    history_plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(history_plot_path)
    plt.close()
    
    print(f"训练完成！模型已保存为 {model_save_path}")
    
    return model, tokenizer, node_types, relations


def balance_training_data(training_data, relation_counts):
    """平衡训练数据，确保各关系类型有足够的样本"""
    # 确定目标样本数 - 使每种关系至少有200个样本
    target_count = max(200, min(relation_counts.values()) * 3)
    
    # 为每种关系创建索引
    relation_indices = {}
    for i, item in enumerate(training_data):
        relation = item["graph"]["edges"][0]["relation"]
        if relation not in relation_indices:
            relation_indices[relation] = []
        relation_indices[relation].append(i)
    
    # 创建平衡后的数据集
    balanced_data = list(training_data)  # 复制原始数据
    
    # 对低频关系进行数据增强
    for relation, count in relation_counts.items():
        if count < target_count:
            # 需要增加的样本数量
            num_to_add = target_count - count
            
            # 可用的索引
            available_indices = relation_indices[relation]
            
            # 通过复制和轻微变化现有样本来增强数据
            for _ in range(num_to_add):
                # 随机选择一个样本
                idx = np.random.choice(available_indices)
                item = training_data[idx]
                
                # 创建变体（这里只是简单复制，实际应用中可以添加文本变换）
                new_item = augment_sample(item)
                balanced_data.append(new_item)
    
    return balanced_data


def augment_sample(item):
    """对样本进行数据增强，生成变体"""
    import copy
    new_item = copy.deepcopy(item)
    
    # 获取原始问题和实体
    question = new_item["question"]
    entity_value = new_item["graph"]["nodes"]["n0"]["value"]
    
    # 简单的增强方法：在问题前添加引导词，或者调整标点
    prefixes = ["请问", "我想知道", "能告诉我", "麻烦问一下", ""]
    suffixes = ["？", "?", ""]
    
    # 移除可能的已有前缀和后缀
    for prefix in prefixes:
        if question.startswith(prefix) and prefix:
            question = question[len(prefix):]
            break
    
    for suffix in suffixes:
        if question.endswith(suffix) and suffix:
            question = question[:-len(suffix)]
            break
    
    # 添加新的前缀和后缀
    new_prefix = np.random.choice(prefixes)
    new_suffix = np.random.choice(suffixes)
    new_question = f"{new_prefix}{question}{new_suffix}"
    
    new_item["question"] = new_question
    
    return new_item


def get_sample_weights(examples):
    """为样本创建权重，使得低频关系类型被更频繁采样"""
    relation_counts = {}
    
    # 计算每种关系的频率
    for item in examples:
        relation = item["graph"]["edges"][0]["relation"]
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    
    # 计算关系的权重 - 与频率成反比
    relation_weights = {}
    total_samples = len(examples)
    num_relations = len(relation_counts)
    
    for relation, count in relation_counts.items():
        relation_weights[relation] = total_samples / (count * num_relations)
    
    # 为每个样本分配权重
    weights = []
    for item in examples:
        relation = item["graph"]["edges"][0]["relation"]
        weights.append(relation_weights[relation])
    
    return weights


if __name__ == "__main__":
    train_graph_parser()
