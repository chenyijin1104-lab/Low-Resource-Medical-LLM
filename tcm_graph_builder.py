import networkx as nx

def build_tcm_graph():
    print("🕸️ 正在内存中构建《伤寒杂病论》知识图谱...")
    G = nx.DiGraph() # 初始化一个有向图

    # ==========================================
    # 1. 导入实体节点 (Nodes)
    # ==========================================
    # 证型
    G.add_node("太阳中风", type="Syndrome")
    # 症状
    G.add_nodes_from(["发热", "恶风", "汗出", "脉缓", "剧烈胸痛", "呼吸困难"], type="Symptom")
    # 方剂
    G.add_node("桂枝汤", type="Formula")
    # 中药
    G.add_nodes_from(["桂枝", "白芍", "炙甘草", "生姜", "大枣"], type="Herb")

    # ==========================================
    # 2. 导入逻辑连线 (Edges / Relationships)
    # ==========================================
    # 证型 -> 症状 (has_symptom)
    G.add_edges_from([
        ("太阳中风", "发热"), ("太阳中风", "恶风"), 
        ("太阳中风", "汗出"), ("太阳中风", "脉缓")
    ], relation="has_symptom")

    # 证型 -> 方剂 (treats_with)
    G.add_edge("太阳中风", "桂枝汤", relation="treats_with")

    # 方剂 -> 中药 (contains_herb)
    G.add_edges_from([
        ("桂枝汤", "桂枝"), ("桂枝汤", "白芍"), 
        ("桂枝汤", "炙甘草"), ("桂枝汤", "生姜"), ("桂枝汤", "大枣")
    ], relation="contains_herb")
    
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
    
    # 模拟一次临床问诊
    patient_symptoms = ["发热", "恶风", "汗出"]
    multi_hop_reasoning(tcm_graph, patient_symptoms)
