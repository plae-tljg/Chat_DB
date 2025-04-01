import json
import random
from collections import defaultdict

# 加载已有的数据作为参考
def load_existing_data():
    with open("data/graph_parser/training_data.json", "r", encoding="utf-8") as f:
        return json.load(f)

# 加载关系映射
def load_relations_mapping():
    with open("data/graph_parser/relations_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)

# 加载节点类型
def load_node_types():
    with open("data/graph_parser/node_types.json", "r", encoding="utf-8") as f:
        return json.load(f)

# 大学列表
universities = [
    "清华大学", "北京大学", "复旦大学", "上海交通大学", "浙江大学", "南京大学", "武汉大学", 
    "中山大学", "哈尔滨工业大学", "西安交通大学", "中国科学技术大学", "同济大学", "南开大学", 
    "天津大学", "华中科技大学", "四川大学", "厦门大学", "山东大学", "吉林大学", "大连理工大学",
    "北京航空航天大学", "北京理工大学", "北京师范大学", "中国人民大学", "东南大学"
]

# 为每种关系生成问题模板
relation_templates = {
    "has_phone": [
        "{university}的电话号码是多少？",
        "如何通过电话联系{university}？",
        "{university}联系电话是什么？",
        "我想打电话给{university}",
        "我需要知道{university}的联系电话",
        "请问{university}的电话怎么找？",
        "{university}电话多少",
        "{university}有什么联系方式？",
        "想打电话联系{university}",
        "{university}的总机电话是多少"
    ],
    "located_in": [
        "{university}在哪里？",
        "{university}的校址在哪？",
        "请告诉我{university}的地址",
        "{university}坐落于哪个城市？",
        "怎么去{university}？它的地址是？",
        "{university}的校园位置在哪？",
        "{university}位于什么地方？",
        "{university}的地理位置是哪里？",
        "到{university}怎么走？",
        "{university}在哪个区？"
    ],
    "has_president": [
        "{university}现任校长是谁？",
        "谁担任{university}校长？",
        "{university}校长的名字是？",
        "{university}的领导者是谁",
        "谁是{university}的校长？",
        "{university}现在的校长是谁？",
        "谁负责管理{university}？",
        "{university}校长是哪位？",
        "请问谁是{university}的现任校长？",
        "{university}是由谁领导的？"
    ],
    "has_website": [
        "{university}的网站链接是什么？",
        "如何进入{university}的官方网站？",
        "{university}网址是多少",
        "请问{university}的网站是什么",
        "如何在网上找到{university}？",
        "{university}的官网地址",
        "我想浏览{university}的网站",
        "{university}官方网站的网址是？",
        "怎样才能访问{university}的网站？",
        "{university}的主页是什么？"
    ],
    "has_tuition": [
        "在{university}就读一年要多少学费？",
        "{university}的收费标准是什么？",
        "{university}本科生一年学费",
        "{university}的学费是多少钱一年？",
        "{university}收费贵吗？具体多少？",
        "就读{university}需要多少学费？",
        "请问{university}的学费标准是什么？",
        "{university}的研究生学费是多少？",
        "{university}国际生的学费是多少？",
        "{university}一学期的学费是多少？"
    ],
    "has_motto": [
        "{university}的校训是什么？",
        "{university}的校训内容",
        "{university}有什么教育理念或校训？",
        "请告诉我{university}的校训",
        "{university}的校训用英语怎么说？",
        "{university}的座右铭是什么？",
        "谁提出了{university}的校训？",
        "{university}的校训有什么含义？",
        "{university}的精神理念是什么？",
        "{university}校训体现了什么价值观？"
    ],
    "founded_in": [
        "{university}什么时候建立的？",
        "{university}创办于哪一年？",
        "{university}已经有多少年历史了？它成立于哪一年？",
        "{university}的建校时间是？",
        "{university}是哪一年开始招生的？",
        "{university}的历史可以追溯到哪一年？",
        "{university}的创始人是谁？创建于何时？",
        "{university}的成立年份是？",
        "{university}已有多少年历史？",
        "{university}的建校日期是什么时候？"
    ],
    "has_student_count": [
        "{university}有多少在校生？",
        "{university}的学生总人数是多少？",
        "{university}在校生规模有多大？",
        "{university}目前的学生数量是多少？",
        "{university}每年招收多少新生？",
        "{university}的学生总数大约是多少？",
        "{university}有多少名学生？",
        "{university}的本科生和研究生总数是多少？",
        "{university}有多少中国学生和国际学生？",
        "{university}的师生比例是多少？学生总数是？"
    ],
    "has_facility": [
        "{university}的图书馆在哪？",
        "{university}的体育馆位置在哪里？",
        "{university}的食堂有几个？",
        "{university}有什么著名的建筑？",
        "{university}的实验室设施如何？",
        "{university}都有哪些教学楼？",
        "{university}的运动场在哪里？",
        "{university}的学生宿舍条件如何？",
        "{university}的医务室在哪？",
        "{university}的会堂在什么位置？"
    ],
    "has_admission_score": [
        "{university}今年的录取分数线是多少？",
        "{university}理科分数线",
        "{university}文科最低录取分是多少？",
        "{university}各省份的录取分数线是多少？",
        "{university}的最低录取标准是什么？",
        "想上{university}需要考多少分？",
        "{university}去年的分数线是多少？",
        "{university}在北京的录取分数线是多少？",
        "{university}的自主招生分数线是多少？",
        "{university}艺术类专业分数线是多少？"
    ],
    "has_international_program": [
        "{university}接收留学生吗？",
        "外国学生可以申请{university}吗？",
        "{university}有英语授课的课程吗？",
        "{university}的国际交流项目有哪些？",
        "{university}与哪些国外大学有合作？",
        "{university}的留学生比例是多少？",
        "{university}有哪些国际合作办学项目？",
        "{university}的国际学生需要什么条件？",
        "{university}提供留学生奖学金吗？",
        "{university}的留学生来自哪些国家？"
    ],
    "predecessor": [
        "{university}的前身是什么学校？",
        "{university}是由哪些学校合并而成的？",
        "{university}最早叫什么名字？",
        "{university}的历史渊源是什么？",
        "{university}是从什么机构发展而来的？",
        "{university}在建校初期是什么样子的？",
        "{university}的前身最初是在哪一年成立的？",
        "{university}历史上经历了哪些更名？",
        "{university}是如何演变成现在的规模的？",
        "{university}最初的校址在哪里？"
    ],
    "nearest_metro": [
        "{university}的地铁站是哪一站？",
        "去{university}最近的地铁站是什么？",
        "乘坐地铁去{university}应该在哪一站下车？",
        "{university}附近有地铁站吗？",
        "从市中心乘地铁去{university}怎么走？",
        "距离{university}最近的公共交通是什么？",
        "{university}校门口有地铁站吗？",
        "去{university}乘几号线地铁最方便？",
        "{university}东门最近的地铁站是哪个？",
        "从地铁站到{university}步行需要多久？"
    ],
    "anniversary_date": [
        "{university}的校庆日是哪一天？",
        "{university}每年的校庆在什么时候举行？",
        "{university}的周年纪念日是几月几日？",
        "{university}的校庆活动通常在什么时候？",
        "{university}的建校纪念日是哪天？",
        "{university}今年要举行多少周年校庆？",
        "{university}的校庆通常会持续几天？",
        "{university}的下一次重要校庆是什么时候？",
        "{university}校庆日有什么特别的活动？",
        "{university}的历史上最重要的校庆是哪一次？"
    ],
    "academician_count": [
        "{university}的院士人数是多少？",
        "{university}有多少两院院士？",
        "{university}拥有多少名科学院院士？",
        "{university}的院士团队有多大？",
        "{university}工程院院士有几位？",
        "{university}科学院院士有几位？",
        "{university}有哪些著名的院士？",
        "{university}近年来新增了多少院士？",
        "{university}院士在全国排名如何？",
        "{university}外籍院士有多少人？"
    ],
    "library_volume": [
        "{university}的图书馆藏书量是多少？",
        "{university}图书馆有多少册图书？",
        "{university}图书馆的规模有多大？",
        "{university}拥有多少电子图书资源？",
        "{university}的文献资源数量是多少？",
        "{university}的图书馆是全国第几大？",
        "{university}图书馆每年新增多少册图书？",
        "{university}的珍本书籍有多少？",
        "{university}图书馆订阅了多少种期刊？",
        "{university}的图书馆有哪些特色馆藏？"
    ],
    "party_secretary": [
        "{university}的现任党委书记是谁？",
        "谁是{university}的党委书记？",
        "{university}党委主要负责人是谁？",
        "{university}的党委书记叫什么名字？",
        "谁担任{university}党委书记职务？",
        "{university}目前的党委书记是哪位？",
        "{university}的党委领导是谁？",
        "谁是{university}党委的一把手？",
        "{university}党委书记上任时间是什么时候？",
        "{university}党委书记有什么主要成就？"
    ],
    "offers_major": [
        "{university}有哪些王牌专业？",
        "{university}的优势学科有哪些？",
        "{university}最好的专业是什么？",
        "{university}都开设了哪些专业？",
        "{university}的特色学科有哪些？",
        "{university}最受欢迎的专业是什么？",
        "{university}的哪些专业全国排名靠前？",
        "{university}的王牌学院是哪个？",
        "{university}的专业设置情况如何？",
        "{university}的热门专业有哪些？"
    ],
    "has_quality": [
        "{university}的计算机专业好吗？",
        "{university}金融系怎么样？",
        "{university}的医学院水平如何？",
        "{university}的法学院怎么样？",
        "{university}工科实力如何？",
        "{university}的建筑系好不好？",
        "{university}的外语专业水平高吗？",
        "{university}的艺术类专业强不强？",
        "{university}的教育学院有特色吗？",
        "{university}的新闻传播学院口碑如何？"
    ]
}

# 生成不同主体节点和查询节点类型的映射
relation_node_types = {
    "has_phone": {"main_type": "university", "query_type": "phone"},
    "located_in": {"main_type": "university", "query_type": "location"},
    "has_president": {"main_type": "university", "query_type": "person"},
    "has_website": {"main_type": "university", "query_type": "website"},
    "has_tuition": {"main_type": "university", "query_type": "tuition"},
    "has_motto": {"main_type": "university", "query_type": "motto"},
    "founded_in": {"main_type": "university", "query_type": "year"},
    "has_student_count": {"main_type": "university", "query_type": "student_count"},
    "has_admission_score": {"main_type": "university", "query_type": "admission_score"},
    "has_international_program": {"main_type": "university", "query_type": "international_program"},
    "predecessor": {"main_type": "university", "query_type": "university"},
    "nearest_metro": {"main_type": "university", "query_type": "metro_station"},
    "anniversary_date": {"main_type": "university", "query_type": "date"},
    "academician_count": {"main_type": "university", "query_type": "number"},
    "library_volume": {"main_type": "university", "query_type": "number"},
    "party_secretary": {"main_type": "university", "query_type": "person"},
    "offers_major": {"main_type": "university", "query_type": "major_list"}
}

# 针对特殊情况的node_types_complex定义
complex_relations = {
    "has_facility": {
        "facilities": [
            {"name": "图书馆", "type": "library"},
            {"name": "体育馆", "type": "sports_center"},
            {"name": "食堂", "type": "canteen"},
            {"name": "实验室", "type": "laboratory"},
            {"name": "宿舍", "type": "dormitory"}
        ],
        "query_type": "location"
    },
    "has_quality": {
        "majors": [
            "计算机科学", "金融学", "医学", "法学", "工程学", 
            "建筑学", "外语", "艺术", "教育学", "新闻传播学",
            "物理学", "化学", "生物学", "数学", "历史学"
        ],
        "query_type": "quality"
    }
}

def generate_training_data(samples_per_relation=10):
    training_data = []
    
    # 对每种关系类型生成样本
    for relation, templates in relation_templates.items():
        for _ in range(samples_per_relation):
            university = random.choice(universities)
            template = random.choice(templates)
            question = template.format(university=university)
            
            # 为复杂关系创建特殊图结构
            if relation == "has_facility":
                # 对于设施类关系，随机选择一种设施
                facility = random.choice(complex_relations["has_facility"]["facilities"])
                facility_name = f"{university}{facility['name']}"
                facility_type = facility["type"]
                query_type = complex_relations["has_facility"]["query_type"]
                
                graph = {
                    "nodes": {
                        "n0": {"type": "university", "value": university},
                        "n1": {"type": facility_type, "value": facility_name},
                        "n2": {"type": query_type, "value": "no_info"}
                    },
                    "edges": [
                        {"from": "n0", "to": "n1", "relation": "has_facility"},
                        {"from": "n1", "to": "n2", "relation": "located_at"}
                    ],
                    "query_node": "n2"
                }
            elif relation == "has_quality":
                # 对于专业质量关系，随机选择一个专业
                major = random.choice(complex_relations["has_quality"]["majors"])
                query_type = complex_relations["has_quality"]["query_type"]
                
                graph = {
                    "nodes": {
                        "n0": {"type": "university", "value": university},
                        "n1": {"type": "major", "value": major},
                        "n2": {"type": query_type, "value": "no_info"}
                    },
                    "edges": [
                        {"from": "n0", "to": "n1", "relation": "offers_major"},
                        {"from": "n1", "to": "n2", "relation": "has_quality"}
                    ],
                    "query_node": "n2"
                }
            else:
                # 普通关系的图结构
                main_type = relation_node_types[relation]["main_type"]
                query_type = relation_node_types[relation]["query_type"]
                
                graph = {
                    "nodes": {
                        "n0": {"type": main_type, "value": university},
                        "n1": {"type": query_type, "value": "no_info"}
                    },
                    "edges": [
                        {"from": "n0", "to": "n1", "relation": relation}
                    ],
                    "query_node": "n1"
                }
            
            training_data.append({
                "question": question,
                "graph": graph
            })
    
    return training_data

def main():
    # 生成每种关系10个样本
    training_data = generate_training_data(samples_per_relation=20)
    
    # 分析每种关系的样本数量
    relation_counts = defaultdict(int)
    for item in training_data:
        for edge in item["graph"]["edges"]:
            relation_counts[edge["relation"]] += 1
    
    print("生成的训练数据关系分布:")
    for relation, count in relation_counts.items():
        print(f"{relation}: {count}个样本")
    
    print(f"总样本数: {len(training_data)}")
    
    # 保存数据
    with open("data/graph_parser/enhanced_training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=4)
    
    print("增强训练数据已保存到 data/graph_parser/enhanced_training_data.json")
    
    # 更新原始训练数据文件
    with open("data/graph_parser/training_data.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=4)
    
    print("训练数据已更新到 data/graph_parser/training_data.json")

if __name__ == "__main__":
    main() 