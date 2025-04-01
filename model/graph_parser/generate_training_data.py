import json
import random
from collections import defaultdict

# 加载知识库数据
with open("data/graph_parser/common/university_knowledge_base.json", "r", encoding="utf-8") as f:
    kb_data = json.load(f)
# 大学列表
universities = list(kb_data.keys())

# 加载关系配置
with open("data/graph_parser/common/relations_config.json", "r", encoding="utf-8") as f:
    relations_config = json.load(f)

# 从配置文件中获取问题模板
relation_templates = {}
for relation, config in relations_config["relations"].items():
    if "question_templates" in config:
        relation_templates[relation] = config["question_templates"]

# 从配置文件中获取节点类型映射
relation_node_types = {}
for relation, config in relations_config["relations"].items():
    if "node_types" in config:
        relation_node_types[relation] = config["node_types"]

# 从配置文件中获取复杂关系配置
complex_relations = {}
for relation, config in relations_config["relations"].items():
    if "complex_config" in config:
        complex_relations[relation] = config["complex_config"]

def get_real_value_for_field(university, field):
    """尝试从知识库中获取真实值"""
    if university in kb_data:
        # 处理嵌套字段
        if "." in field:
            parts = field.split(".", 1)
            if parts[0] in kb_data[university] and isinstance(kb_data[university][parts[0]], dict):
                nested_data = kb_data[university][parts[0]]
                if parts[1] in nested_data:
                    return nested_data[parts[1]]
        # 处理普通字段
        elif field in kb_data[university]:
            return kb_data[university][field]
    return "no_info"

def generate_training_data(samples_per_relation=10, use_real_values=True):
    training_data = []
    
    # 对每种关系类型生成样本
    for relation, templates in relation_templates.items():
        for _ in range(samples_per_relation):
            university = random.choice(universities)
            template = random.choice(templates)
            question = template.format(university=university)
            
            # 获取该关系对应的数据库字段
            db_field = None
            if relation in relations_config["relations"]:
                db_field = relations_config["relations"][relation].get("db_field")
            
            # 检查是否为复杂关系
            if relation in complex_relations:
                # 获取复杂关系配置
                complex_config = complex_relations[relation]
                
                # 根据复杂关系类型创建特殊图结构
                if relation == "has_facility" and "facilities" in complex_config:
                    # 对于设施类关系，随机选择一种设施
                    facility = random.choice(complex_config["facilities"])
                    facility_name = f"{university}{facility['name']}"
                    facility_type = facility["type"]
                    query_type = complex_config["query_type"]
                    
                    # 创建查询图
                    graph = {
                        "nodes": {
                            "n0": {"type": "university", "value": university},
                            "n1": {"type": facility_type, "value": facility_name},
                            "n2": {"type": query_type, "value": "no_info"}
                        },
                        "edges": [
                            {"from": "n0", "to": "n1", "relation": relation},
                            {"from": "n1", "to": "n2", "relation": "located_at"}
                        ],
                        "query_node": "n2"
                    }
                else:
                    # 对于其他复杂关系，使用通用处理方式
                    main_type = relation_node_types[relation]["main_type"]
                    query_type = relation_node_types[relation]["query_type"]
                    
                    # 获取真实值(如果可能)
                    query_value = "no_info"
                    if use_real_values and db_field:
                        query_value = get_real_value_for_field(university, db_field)
                    
                    graph = {
                        "nodes": {
                            "n0": {"type": main_type, "value": university},
                            "n1": {"type": query_type, "value": query_value}
                        },
                        "edges": [
                            {"from": "n0", "to": "n1", "relation": relation}
                        ],
                        "query_node": "n1"
                    }
            else:
                # 普通关系的图结构
                main_type = relation_node_types[relation]["main_type"]
                query_type = relation_node_types[relation]["query_type"]
                
                # 获取真实值(如果可能)
                query_value = "no_info"
                if use_real_values and db_field:
                    query_value = get_real_value_for_field(university, db_field)
                
                graph = {
                    "nodes": {
                        "n0": {"type": main_type, "value": university},
                        "n1": {"type": query_type, "value": query_value}
                    },
                    "edges": [
                        {"from": "n0", "to": "n1", "relation": relation}
                    ],
                    "query_node": "n1"
                }
            
            # 生成正向和反向训练样本
            # 正向: 问题 -> 查询图(结果未知)
            forward_sample = {
                "question": question,
                "graph": graph.copy()
            }
            # 确保查询节点值为空
            forward_sample["graph"]["nodes"][forward_sample["graph"]["query_node"]]["value"] = "no_info"
            training_data.append(forward_sample)
            
            # 只有当有真实值且不是"no_info"时，才添加反向样本
            if use_real_values and graph["nodes"][graph["query_node"]]["value"] != "no_info":
                # 反向: 完整图 -> 生成问题(用于训练问答生成)
                backward_sample = {
                    "graph": graph.copy(),
                    "question": question
                }
                training_data.append(backward_sample)
    
    return training_data

def main():
    # 生成每种关系20个样本
    training_data = generate_training_data(samples_per_relation=20, use_real_values=True)
    
    # 分析训练数据
    relation_counts = defaultdict(int)
    forward_samples = 0
    backward_samples = 0
    complete_samples = 0
    
    for item in training_data:
        # 检查是正向还是反向样本
        if "question" in item and "graph" in item:
            query_node = item["graph"]["query_node"]
            if item["graph"]["nodes"][query_node]["value"] == "no_info":
                forward_samples += 1
            else:
                backward_samples += 1
                
            # 统计关系类型
            for edge in item["graph"]["edges"]:
                relation_counts[edge["relation"]] += 1
                
            # 检查是否有完整信息的样本
            if all(node["value"] != "no_info" for node_id, node in item["graph"]["nodes"].items()):
                complete_samples += 1
    
    # 打印分析结果
    print("\n生成的训练数据分析:")
    print(f"总样本数: {len(training_data)}")
    print(f"正向样本(问题→图): {forward_samples}")
    print(f"反向样本(图→问题): {backward_samples}")
    print(f"完整信息样本: {complete_samples}")
    
    print("\n关系分布:")
    for relation, count in relation_counts.items():
        print(f"{relation}: {count}个样本")
        
    # 保存训练数据
    with open("data/graph_parser/training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=4)
    
    print("\n训练数据已更新到 data/graph_parser/training_data.json")
    
    # 生成一部分纯输入和纯标签数据，用于模型微调
    input_only = [{"question": item["question"]} for item in training_data if "question" in item and "graph" in item and item["graph"]["nodes"][item["graph"]["query_node"]]["value"] == "no_info"]
    label_only = [{"graph": item["graph"]} for item in training_data if "question" in item and "graph" in item and item["graph"]["nodes"][item["graph"]["query_node"]]["value"] == "no_info"]
    
    # 保存独立的输入和标签数据
    with open("data/graph_parser/input_questions.json", "w", encoding="utf-8") as f:
        json.dump(input_only, f, ensure_ascii=False, indent=2)
        
    with open("data/graph_parser/output_graphs.json", "w", encoding="utf-8") as f:
        json.dump(label_only, f, ensure_ascii=False, indent=2)
    
    print("还分别生成了独立的输入问题和输出图文件，用于模型微调。")

if __name__ == "__main__":
    main() 