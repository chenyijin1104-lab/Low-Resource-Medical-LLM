import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import BertConfig, BertModel, BertTokenizer
import chromadb
from chromadb.utils import embedding_functions

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
# 1. 理智轨道：医学 RAG 检索器 (自带防御机制)
# ==========================================
class MedicalRAGRetriever:
    def __init__(self, db_path="./medical_db", collection_name="sino_western_medical_knowledge"):
        self.bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-zh-v1.5")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.bge_ef)
        
        if self.collection.count() == 0:
            self._auto_inject_medical_knowledge()

    def _auto_inject_medical_knowledge(self):
        self.collection.upsert(
            documents=[
                "自发性气胸：好发于瘦高体型青年男性。常在剧烈运动（如打篮球）、剧烈咳嗽时突发单侧胸部针刺样剧痛，伴呼吸困难、大汗淋漓。查体可见气管向健侧偏位。",
                "大叶性肺炎：由肺炎链球菌引起，典型症状为起病急骤，高热、寒战、咳嗽、咳铁锈色痰，胸痛。听诊闻及湿啰音。",
                "太阳病，发热汗出，恶风，脉缓者，名为中风。",
                "少阳病，口苦，咽干，目眩也。常伴有往来寒热，胸胁苦满，心烦喜呕。"
            ],
            metadatas=[
                {"source": "《现代急诊医学指南》", "category": "西医", "disease": "气胸"},
                {"source": "《内科学》", "category": "西医", "disease": "大叶性肺炎"},
                {"source": "《伤寒论》- 太阳病篇", "category": "中医", "disease": "太阳中风"},
                {"source": "《伤寒论》- 少阳病篇", "category": "中医", "disease": "少阳病"}
            ],
            ids=["doc_med_01", "doc_med_02", "doc_tcm_01", "doc_tcm_02"]
        )

    def get_clinical_advice(self, patient_complaint, top_k=1):
        results = self.collection.query(query_texts=[patient_complaint], n_results=top_k)
        advice_list = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                advice_list.append({
                    "source": results['metadatas'][0][i]['source'],
                    "disease": results['metadatas'][0][i]['disease'],
                    "content": results['documents'][0][i]
                })
        return advice_list

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
# 3. 顶层仲裁法庭：Logit 动态校准机制
# ==========================================
def dual_track_diagnosis(text_content, xray_path, model_path="best_medical_model_sota.pth", rag_boost_weight=5.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*65)
    print("🏥 【中西医双轨制医疗大脑】联合会诊启动")
    print("="*65)
    print(f"📄 病患主诉: {text_content}")
    print("-" * 65)

    # --- 系统 2: RAG 翻书检索 ---
    print("📚 [理智轨道] 正在连接 ChromaDB 医学典籍库...")
    retriever = MedicalRAGRetriever()
    advices = retriever.get_clinical_advice(text_content, top_k=1)
    
    rag_disease = None
    if advices:
        rag_disease = advices[0]['disease']
        print(f"🎯 知识库命中: {advices[0]['source']} 指出符合【{rag_disease}】指征。")
    
    # --- 系统 1: 多模态看片 ---
    print("👁️  [视觉轨道] 启动 Cross-Attention 多模态网络看片...")
    inference_model = MedicalMultimodalModel(num_classes=10).to(device)
    try:
        inference_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"\n❌ 错误：找不到模型权重文件 '{model_path}'。请确保你在当前目录下运行过训练脚本。")
        return
        
    inference_model.eval() 
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
    encoding = tokenizer(text_content, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    
    try:
        img = Image.open(xray_path).convert('L').resize((256, 256))
    except FileNotFoundError:
        print(f"\n❌ 错误：找不到 X 光片 '{xray_path}'。请确保 simulated_xrays 文件夹存在。")
        return
        
    img_tensor = (torch.tensor(np.array(img, dtype=np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0 - 0.5) / 0.5
    
    with torch.no_grad(): 
        with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            logits = inference_model(img_tensor.to(device), encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
            raw_probabilities = F.softmax(logits, dim=1).flatten().cpu().numpy()

    # --- 仲裁介入: RAG 强行拉升特定疾病权重 ---
    print("⚖️  [仲裁中心] 执行 Logit 动态校准机制...")
    if rag_disease in REVERSE_DISEASE_MAP:
        target_idx = REVERSE_DISEASE_MAP[rag_disease]
        logits[0, target_idx] += rag_boost_weight
        print(f"💉 知识库干预生效：强行向【{rag_disease}】维度注入置信惩罚 (Boost: +{rag_boost_weight})！")
    
    final_probabilities = F.softmax(logits, dim=1).flatten().cpu().numpy()

    print("-" * 65)
    print(f"{'疾病名称':<8} | {'纯视觉直觉 (底层幻觉)':<18} | {'双轨仲裁后最终概率':<18}")
    print("-" * 65)
    
    for i in range(10):
        print(f"{DISEASE_MAP[i]:<10} | {raw_probabilities[i]*100:>15.2f}% | {final_probabilities[i]*100:>15.2f}%")
        
    top_class_raw = np.argmax(raw_probabilities)
    top_class_final = np.argmax(final_probabilities)
    
    print("=" * 65)
    print(f"❌ 盲目诊断 (仅多模态): 疑似【{DISEASE_MAP[top_class_raw]}】")
    print(f"✅ 循证诊断 (多模态+RAG): 高度确诊【{DISEASE_MAP[top_class_final]}】")
    print("=" * 65)

if __name__ == "__main__":
    test_text = "患者瘦高体型，打篮球时突感右胸针刺样剧痛，伴大汗淋漓，叩诊鼓音，气管向健侧偏。"
    test_image_path = "simulated_xrays/img_7.jpg" 
    
    dual_track_diagnosis(text_content=test_text, xray_path=test_image_path)
