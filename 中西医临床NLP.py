import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import BertConfig, BertModel, BertTokenizer
import networkx as nx
import torchvision.models as models # 【新增】：引入视觉顶尖模型库

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
# 1. 理智轨道：中医知识图谱推理引擎 (GraphRAG) - 数据解耦版
# ==========================================
class MedicalGraphEngine:
    def __init__(self, json_path="data/kg_data/tcm_knowledge.json"):
        self.G = nx.DiGraph()
        self.json_path = json_path
        self._build_sino_western_knowledge_graph()

    def _build_sino_western_knowledge_graph(self):
        """
        从底层 JSON 数据库动态渲染中西医结合星网
        """
        import json
        if not os.path.exists(self.json_path):
            print(f"❌ 严重错误：未找到底层数据库 {self.json_path}")
            return
            
        with open(self.json_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
            
        for item in knowledge_base["syndromes"]:
            syndrome = item["name"]
            western_disease = item["western_disease"]
            symptoms = item["symptoms"]
            formula = item["formula"]
            herbs = item["herbs"]
            
            # 构建跨界桥梁
            self.G.add_node(syndrome, type="TCMSyndrome")
            self.G.add_node(western_disease, type="WesternDisease")
            self.G.add_edge(western_disease, syndrome, relation="mapped_to_syndrome")
            
            # 构建症状、方剂、药材拓扑图
            for sym in symptoms:
                self.G.add_node(sym, type="Symptom")
                self.G.add_edge(syndrome, sym, relation="has_symptom")
            self.G.add_node(formula, type="Formula")
            self.G.add_edge(syndrome, formula, relation="treats_with")
            for herb in herbs:
                self.G.add_node(herb, type="Herb")
                self.G.add_edge(formula, herb, relation="contains_herb")

    def graph_reasoning(self, text_content):
        # 关键词命中检测
        hit_symptoms = [node for node, data in self.G.nodes(data=True) if data.get('type') == 'Symptom' and node in text_content]
                
        if not hit_symptoms:
            return None, None, "❌ 图谱未在主诉中抽离出核心中医体征"

        # 【第一跳】：聚类中医证型
        matched_syndromes = []
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'TCMSyndrome':
                expected_symptoms = [tgt for _, tgt, rel in self.G.out_edges(node, data='relation') if rel == 'has_symptom']
                score = sum(1 for sym in hit_symptoms if sym in expected_symptoms)
                if score > 0: # 命中至少一个症状即纳入疑似
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
# 2. 直觉轨道：多模态视觉底座 (v4.0 ResNet 升级版)
# ==========================================
class MedicalMultimodalModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 【微创手术 1】：引入预训练的 ResNet-50，截断全局池化和全连接层
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # 【微创手术 2】：硬件防爆机制，冻结前 80% 的网络，只放开最深层的特征捕捉能力
        for param in self.cnn[:-1].parameters():
            param.requires_grad = False
            
        # 【微创手术 3】：特征降维打击 (ResNet 2048维 -> 多模态 256维)
        self.img_proj = nn.Linear(2048, 256) 
        
        # 文本底座保持完全一致
        self.bert = BertModel(BertConfig(vocab_size=21128, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024))
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 128), nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(128, num_classes)
        )
        
    def forward(self, images, input_ids, attention_mask):
        # 动态通道拓展：将单通道灰度图复制为 3 通道以适配 ResNet
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
            
        # 输入 256x256 -> 经过 ResNet 变成 (Batch, 2048, 8, 8)
        img_f = self.cnn(images) 
        
        # 展平空间维度，转换为 Cross-Attention 识别的序列 (Batch, Sequence=64, Dim=2048)
        img_seq = img_f.view(img_f.size(0), 2048, -1).permute(0, 2, 1) 
        img_seq = F.relu(self.img_proj(img_seq)) # 降维至 (Batch, 64, 256)
        
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
    print("🏥 【中西医双轨制医疗大脑 v4.0 - ResNet全域网络】联合会诊启动")
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
    test_text = "患者近期消瘦明显，伴有干咳、夜间盗汗，下午有潮热现象，请求专家会诊。"
    test_image_path = "./data/chest_xray/train/PNEUMONIA/person80_virus_150.jpeg" 
    dual_track_diagnosis(text_content=test_text, xray_path=test_image_path)
