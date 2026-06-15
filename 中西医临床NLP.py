import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import BertConfig, BertModel, BertTokenizer
import networkx as nx

# 禁用 tokenizers 的并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 0. 全局映射表
# ==========================================
DISEASE_MAP = {
    0: "正常", 1: "肺炎", 2: "肺结核", 3: "支气管炎", 4: "肺气肿",
    5: "胸腔积液", 6: "肺癌", 7: "气胸", 8: "肺大泡", 9: "肺心病"
}
REVERSE_DISEASE_MAP = {v: k for k, v in DISEASE_MAP.items()}

# ==========================================
# 1. 理智轨道：中医知识图谱推理引擎 (GraphRAG) - 肺系病深度扩展版
# ==========================================
class MedicalGraphEngine:
    def __init__(self):
        self.G = nx.DiGraph()
        self._build_sino_western_knowledge_graph()

    def _build_sino_western_knowledge_graph(self):
        """
        在内存中编织一张【极深厚】的中西医结合肺系病知识星网
        """
        # A. 导入西医疾病节点
        self.G.add_nodes_from(["正常", "肺炎", "气胸", "支气管炎", "肺结核", "支气管哮喘", "肺气肿"], type="WesternDisease")
        
        # B. 导入中医证型节点
        self.G.add_nodes_from([
            "太阳中风证", "痰热壅肺证", "肺络受损证", 
            "风寒袭肺证", "肺阴亏耗证", "寒饮伏肺证", "痰浊壅肺证"
        ], type="TCMSyndrome")
        
        # C. 导入临床体征/症状节点 (极大扩充)
        self.G.add_nodes_from([
            "发热", "恶风", "汗出", "脉缓",                  # 太阳中风
            "高热", "咳嗽", "咳吐黄脓痰", "双肺听诊湿啰音",     # 痰热壅肺(肺炎)
            "瘦高体型", "突发单侧胸部剧痛", "呼吸困难", "叩诊鼓音", # 肺络受损(气胸)
            "咳嗽声重", "气急", "咽痒", "咳痰稀白", "鼻塞流清涕",  # 风寒袭肺(支气管炎)
            "干咳", "咳血", "潮热", "盗汗", "消瘦",           # 肺阴亏耗(肺结核)
            "喘息哮鸣", "胸膈满闷", "形寒怕冷", "不能平卧",      # 寒饮伏肺(哮喘)
            "咳逆喘满", "胸部膨满", "咳痰白黏", "动则喘甚"       # 痰浊壅肺(慢阻肺/肺气肿)
        ], type="Symptom")
        
        # D. 导入经方方剂节点
        self.G.add_nodes_from([
            "桂枝汤", "麻杏石甘汤", "百合固金汤加减", 
            "三拗汤", "月华丸", "小青龙汤", "苏子降气汤"
        ], type="Formula")
        
        # E. 导入底层中药材节点
        self.G.add_nodes_from([
            "桂枝", "白芍", "炙甘草", "生姜", "大枣",         
            "麻黄", "苦杏仁", "石膏",                        
            "百合", "生地黄", "熟地黄", "麦冬", "玄参", "贝母",
            "天冬", "山药", "百部", "沙参", "阿胶",          
            "干姜", "细辛", "五味子", "半夏",               
            "紫苏子", "前胡", "厚朴", "肉桂", "当归"         
        ], type="Herb")

        # =========================================================
        # 编织逻辑连线
        # =========================================================
        # 1. 跨界映射 (WesternDisease -> TCMSyndrome)
        self.G.add_edges_from([
            ("肺炎", "痰热壅肺证"), ("气胸", "肺络受损证"),
            ("支气管炎", "风寒袭肺证"), ("肺结核", "肺阴亏耗证"),
            ("支气管哮喘", "寒饮伏肺证"), ("肺气肿", "痰浊壅肺证")
        ], relation="mapped_to_syndrome")

        # 2. 中医辨证 (TCMSyndrome -> Symptom)
        self.G.add_edges_from([
            ("太阳中风证", "发热"), ("太阳中风证", "恶风"), ("太阳中风证", "汗出"), ("太阳中风证", "脉缓"),
            ("痰热壅肺证", "高热"), ("痰热壅肺证", "咳嗽"), ("痰热壅肺证", "咳吐黄脓痰"), ("痰热壅肺证", "双肺听诊湿啰音"),
            ("肺络受损证", "瘦高体型"), ("肺络受损证", "突发单侧胸部剧痛"), ("肺络受损证", "呼吸困难"), ("肺络受损证", "叩诊鼓音"),
            ("风寒袭肺证", "咳嗽声重"), ("风寒袭肺证", "气急"), ("风寒袭肺证", "咽痒"), ("风寒袭肺证", "咳痰稀白"), ("风寒袭肺证", "鼻塞流清涕"),
            ("肺阴亏耗证", "干咳"), ("肺阴亏耗证", "咳血"), ("肺阴亏耗证", "潮热"), ("肺阴亏耗证", "盗汗"), ("肺阴亏耗证", "消瘦"),
            ("寒饮伏肺证", "喘息哮鸣"), ("寒饮伏肺证", "胸膈满闷"), ("寒饮伏肺证", "形寒怕冷"), ("寒饮伏肺证", "不能平卧"),
            ("痰浊壅肺证", "咳逆喘满"), ("痰浊壅肺证", "胸部膨满"), ("痰浊壅肺证", "咳痰白黏"), ("痰浊壅肺证", "动则喘甚")
        ], relation="has_symptom")

        # 3. 确立治法 (TCMSyndrome -> Formula)
        self.G.add_edges_from([
            ("太阳中风证", "桂枝汤"), ("痰热壅肺证", "麻杏石甘汤"), ("肺络受损证", "百合固金汤加减"),
            ("风寒袭肺证", "三拗汤"), ("肺阴亏耗证", "月华丸"), ("寒饮伏肺证", "小青龙汤"), ("痰浊壅肺证", "苏子降气汤")
        ], relation="treats_with")

        # 4. 药方配伍 (Formula -> Herb)
        self.G.add_edges_from([
            ("桂枝汤", "桂枝"), ("桂枝汤", "白芍"), ("桂枝汤", "炙甘草"), ("桂枝汤", "生姜"), ("桂枝汤", "大枣"),
            ("麻杏石甘汤", "麻黄"), ("麻杏石甘汤", "苦杏仁"), ("麻杏石甘汤", "石膏"), ("麻杏石甘汤", "炙甘草"),
            ("百合固金汤加减", "百合"), ("百合固金汤加减", "生地黄"), ("百合固金汤加减", "熟地黄"), ("百合固金汤加减", "贝母"),
            ("三拗汤", "麻黄"), ("三拗汤", "苦杏仁"), ("三拗汤", "炙甘草"),
            ("月华丸", "天冬"), ("月华丸", "麦冬"), ("月华丸", "生地黄"), ("月华丸", "熟地黄"), ("月华丸", "山药"), ("月华丸", "百部"), ("月华丸", "沙参"), ("月华丸", "贝母"), ("月华丸", "阿胶"),
            ("小青龙汤", "麻黄"), ("小青龙汤", "桂枝"), ("小青龙汤", "干姜"), ("小青龙汤", "细辛"), ("小青龙汤", "五味子"), ("小青龙汤", "白芍"), ("小青龙汤", "半夏"), ("小青龙汤", "炙甘草"),
            ("苏子降气汤", "紫苏子"), ("苏子降气汤", "半夏"), ("苏子降气汤", "前胡"), ("苏子降气汤", "厚朴"), ("苏子降气汤", "肉桂"), ("苏子降气汤", "当归"), ("苏子降气汤", "炙甘草")
        ], relation="contains_herb")

    def graph_reasoning(self, text_content):
        # 关键词命中检测
        hit_symptoms = []
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'Symptom' and node in text_content:
                hit_symptoms.append(node)
                
        if not hit_symptoms:
            return None, None, "❌ 图谱未在主诉中抽离出核心中医体征"

        # 【第一跳】：聚类中医证型
        matched_syndromes = []
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'TCMSyndrome':
                expected_symptoms = [tgt for _, tgt, rel in self.G.out_edges(node, data='relation') if rel == 'has_symptom']
                score = sum(1 for sym in hit_symptoms if sym in expected_symptoms)
                if score > 0:
                    matched_syndromes.append((node, score))
                    
        if not matched_syndromes:
            return None, None, "❌ 无法形成有效的证型聚类"
            
        matched_syndromes.sort(key=lambda x: x[1], reverse=True)
        best_syndrome = matched_syndromes[0][0]

        # 【第二跳】：寻找西医跨界连线
        mapped_western_diseases = [src for src, tgt, rel in self.G.in_edges(best_syndrome, data='relation') if rel == 'mapped_to_syndrome']
        target_western_disease = mapped_western_diseases[0] if mapped_western_diseases else None

        # 【第三跳】：推导方剂
        formulas = [tgt for _, tgt, rel in self.G.out_edges(best_syndrome, data='relation') if rel == 'treats_with']
        best_formula = formulas[0] if formulas else "辨证施治"

        # 【第四跳】：展开中药
        herbs = [tgt for _, tgt, rel in self.G.out_edges(best_formula, data='relation') if rel == 'contains_herb']
        
        report = f"💡 [全域图谱多跳推理成功]\n" \
                 f"  ├─ 命中体征: {hit_symptoms}\n" \
                 f"  ├─ 辩证证型: 确立为【{best_syndrome}】\n" \
                 f"  ├─ 对应方剂: 推荐经方【{best_formula}】\n" \
                 f"  └─ 药方配伍: {', '.join(herbs)}"
                 
        return target_western_disease, report

# ==========================================
# 2. 直觉轨道：多模态视觉底座
# ==========================================
class MedicalMultimodalModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((16, 16)) 
        )
        self.img_proj = nn.Linear(64, 256) 
        self.bert = BertModel(BertConfig(vocab_size=21128, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024))
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 128), nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(128, num_classes)
        )
        
    def forward(self, images, input_ids, attention_mask):
        img_f = self.cnn(images) 
        img_seq = F.relu(self.img_proj(img_f.view(img_f.size(0), 64, -1).permute(0, 2, 1))) 
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        txt_pooler = bert_out.pooler_output 
        attn_out, _ = self.cross_attn(query=txt_pooler.unsqueeze(1), key=img_seq, value=img_seq)
        fused_features = torch.cat((attn_out.squeeze(1), txt_pooler), dim=1)
        return self.classifier(fused_features)

# ==========================================
# 3. 顶层仲裁法庭：Logit 动态校准机制 (全域 GraphRAG)
# ==========================================
def dual_track_diagnosis(text_content, xray_path, model_path="./checkpoints/real_medical_model_sota.pth", rag_boost_weight=20.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print("🏥 【中西医双轨制医疗大脑 v3.0 - 全域肺系病知识网络】联合会诊启动")
    print("="*70)
    print(f"📄 病患主诉: {text_content}")
    print("-" * 70)

    # --- 系统 2: 图谱理智轨道 ---
    print("📚 [理智轨道] 检索 NetworkX 深度结构化中西医星网...")
    graph_engine = MedicalGraphEngine()
    target_western_disease, graph_report = graph_engine.graph_reasoning(text_content)
    
    if graph_report:
        print(graph_report)
    
    # --- 系统 1: 多模态视觉直觉轨道 ---
    print("\n👁️  [视觉轨道] 启动 Cross-Attention 多模态网络看片...")
    inference_model = MedicalMultimodalModel(num_classes=10).to(device)
    try:
        inference_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"\n❌ 错误：找不到模型权重文件 '{model_path}'。")
        return
        
    inference_model.eval() 
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
    encoding = tokenizer(text_content, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    
    try:
        img = Image.open(xray_path).convert('L').resize((256, 256))
    except FileNotFoundError:
        print(f"\n❌ 错误：找不到 X 光片 '{xray_path}'。")
        return
        
    img_tensor = (torch.tensor(np.array(img, dtype=np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0 - 0.5) / 0.5
    
    with torch.no_grad(): 
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            logits = inference_model(img_tensor.to(device), encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
            raw_probabilities = F.softmax(logits, dim=1).flatten().cpu().numpy()

    # --- 仲裁介入 ---
    print("\n⚖️  [仲裁中心] 执行 Logit 动态校准机制...")
    if target_western_disease in REVERSE_DISEASE_MAP:
        target_idx = REVERSE_DISEASE_MAP[target_western_disease]
        logits[0, target_idx] += rag_boost_weight
        print(f"💉 图谱铁腕干预生效：跨界对齐成功，强行向西医【{target_western_disease}】维度注入惩罚 (Boost: +{rag_boost_weight})！")
    
    final_probabilities = F.softmax(logits, dim=1).flatten().cpu().numpy()

    print("-" * 70)
    print(f"{'疾病名称':<8} | {'纯视觉直觉 (底层幻觉)':<18} | {'双轨仲裁后最终概率':<18}")
    print("-" * 70)
    
    for i in range(10):
        print(f"{DISEASE_MAP[i]:<10} | {raw_probabilities[i]*100:>15.2f}% | {final_probabilities[i]*100:>15.2f}%")
        
    top_class_raw = np.argmax(raw_probabilities)
    top_class_final = np.argmax(final_probabilities)
    
    print("=" * 70)
    print(f"❌ 盲目诊断 (仅多模态): 疑似【{DISEASE_MAP[top_class_raw]}】")
    print(f"✅ 循证诊断 (多模态+图谱): 高度确诊【{DISEASE_MAP[top_class_final]}】")
    print("=" * 70)

if __name__ == "__main__":
    # ⚠️ 极其苛刻的新测试用例：
    # 患者给的是肺结核的阴虚体征，但由于模型没见过这图，视觉依然可能会乱猜。
    test_text = "患者近期消瘦明显，伴有干咳、夜间盗汗，下午有潮热现象，请求专家会诊。"
    
    # 真实测试影像 (依然沿用你文件夹里的病毒性肺炎，作为干扰项)
    test_image_path = "./data/chest_xray/train/PNEUMONIA/person80_virus_150.jpeg" 
    
    dual_track_diagnosis(text_content=test_text, xray_path=test_image_path)
