import networkx as nx
import json
import os

def build_tcm_graph(json_path="data/kg_data/tcm_knowledge.json"):
    print(f"🕸️ 正在从底层数据库 {json_path} 加载并动态构建专家知识图谱...")
    G = nx.DiGraph() # 初始化一个有向图

    # 确保文件存在
    if not os.path.exists(json_path):
        print(f"❌ 找不到数据文件：{json_path}，请检查路径！")
        return G

    # ==========================================
    # 1. 打开并读取 JSON 知识库 (数据与逻辑剥离)
    # ==========================================
    with open(json_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    # ==========================================
    # 2. 动态遍历字典，自动渲染图谱节点和边
    # ==========================================
    for item in knowledge_base["syndromes"]:
        syndrome = item["name"]
        symptoms = item["symptoms"]
        formula = item["formula"]
        herbs = item["herbs"]

        # 添加【证型】节点
        G.add_node(syndrome, type="Syndrome")
        
        # 添加【症状】节点及连线 (has_symptom)
        for sym in symptoms:
            G.add_node(sym, type="Symptom")
            G.add_edge(syndrome, sym, relation="has_symptom")
        
        # 添加【方剂】节点及连线 (treats_with)
        G.add_node(formula, type="Formula")
        G.add_edge(syndrome, formula, relation="treats_with")
        
        # 添加【中药】节点及连线 (contains_herb)
        for herb in herbs:
            G.add_node(herb, type="Herb")
            G.add_edge(formula, herb, relation="contains_herb")
            
    print(f"✅ 图谱构建完成！当前图谱包含 {G.number_of_nodes()} 个中医实体节点，{G.number_of_edges()} 条逻辑边。")
    return G

def multi_hop_reasoning(G, input_symptoms):
    print(f"\n🔍 启动多跳推理引擎 | 患者主诉症状: {input_symptoms}")
    
    # 第 1 跳：寻找这些症状共同指向的“证型”
    possible_syndromes = []
    for node, data in G.nodes(data=True):
        if data.get('type') == 'Syndrome':
            # 获取该证型的所有症状
            syndrome_symptoms = [target for _, target, rel in G.out_edges(node, data='relation') if rel == 'has_symptom']
            
            # 如果患者的症状在该证型的范围内，且匹配度较高，则纳入考虑
            match_count = sum(1 for sym in input_symptoms if sym in syndrome_symptoms)
            if match_count >= 2: # 设定阈值：至少命中两个核心症状才算疑似
                possible_syndromes.append((node, match_count))
    
    if not possible_syndromes:
        print("❌ 图谱中未找到高度匹配的证型，无法进行下一步推理。")
        return None
        
    # 按匹配度排序，取最可能的证型
    possible_syndromes.sort(key=lambda x: x[1], reverse=True)
    best_syndrome = possible_syndromes[0][0]
    print(f"💡 [第 1 跳结论] 症状溯源成功：高度疑似【{best_syndrome}】")
    
    # 第 2 跳：根据证型推导治疗“方剂”
    formulas = [target for _, target, rel in G.out_edges(best_syndrome, data='relation') if rel == 'treats_with']
    if not formulas:
        return None
        
    best_formula = formulas[0]
    print(f"💊 [第 2 跳结论] 确立治法方剂：推荐使用【{best_formula}】")
    
    # 第 3 跳：展开方剂的“中药”组成
    herbs = [target for _, target, rel in G.out_edges(best_formula, data='relation') if rel == 'contains_herb']
    print(f"🌿 [第 3 跳结论] 自动生成处方：{', '.join(herbs)}")
    
    return best_syndrome, best_formula

if __name__ == "__main__":
    # 实例化图谱
    tcm_graph = build_tcm_graph()
    
    # 模拟一次临床问诊（已替换为测试“痰热郁肺”的特异性症状）
    patient_symptoms = ["喘咳气涌", "胸闷", "痰多色黄黏稠"]
    multi_hop_reasoning(tcm_graph, patient_symptoms)
