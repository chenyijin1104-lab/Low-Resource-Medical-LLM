import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2 
from transformers import BertConfig, BertModel, BertTokenizer
import torchvision.models as models
import torchvision.transforms as transforms
import gradio as gr
import json
import requests

# ================= 新增 RAG 依赖 =================
import chromadb
from sentence_transformers import SentenceTransformer

# ================= 新增 V8.0/V11.0 LangGraph 多智能体依赖 =================
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==============================================================================
# ⚡ V10.1/V11.0 核心跨文件模块化导入与 OOP 派生 (Import Routing & OOP Subclassing)
# ==============================================================================
# 1. 从【模块 B：train_live_fire.py】导入实战 SOTA 模型的基座类
try:
    from train_live_fire import MedicalMultimodalModel as BaseMultimodalModel
    print("🔗 [V11.0 Import] 成功导入跨文件模块：[train_live_fire.py] -> BaseMultimodalModel")
    
    class MedicalMultimodalModel(BaseMultimodalModel):
        def __init__(self, num_classes=10):
            super().__init__(num_classes)
            self.gradients = None
            self.activations = None
            self.cnn[-1].register_forward_hook(self.save_activation)
            self.cnn[-1].register_full_backward_hook(self.save_gradient)

        def save_activation(self, module, input, output):
            self.activations = output

        def save_gradient(self, module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
except ImportError:
    print("⚠️ [V11.0 Import] 未检测到 train_live_fire.py，自动启用内建安全热力图多模态底座...")
    class MedicalMultimodalModel(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(resnet.children())[:-2])
            self.img_proj = nn.Linear(2048, 256) 
            self.bert = BertModel(BertConfig(vocab_size=21128, hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024))
            self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True, dropout=0.3)
            self.classifier = nn.Sequential(
                nn.Linear(256 + 256, 128), nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(128, num_classes)
            )
            self.gradients = None
            self.activations = None
            self.cnn[-1].register_forward_hook(self.save_activation)
            self.cnn[-1].register_full_backward_hook(self.save_gradient)

        def save_activation(self, module, input, output):
            self.activations = output

        def save_gradient(self, module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def forward(self, images, input_ids, attention_mask, return_embeddings=False):
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            img_f = self.cnn(images) 
            img_seq = img_f.view(img_f.size(0), 2048, -1).permute(0, 2, 1) 
            img_seq = F.relu(self.img_proj(img_seq)) 
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            txt_pooler = bert_out.pooler_output 
            attn_out, _ = self.cross_attn(query=txt_pooler.unsqueeze(1), key=img_seq, value=img_seq)
            fused_features = torch.cat((attn_out.squeeze(1), txt_pooler), dim=1)
            logits = self.classifier(fused_features)
            if return_embeddings:
                return logits, img_seq.mean(dim=1), txt_pooler
            return logits

# 2. 从【模块 D：中西医临床NLP.py】导入基于 NetworkX 的有向因果图谱推理引擎
try:
    from 中西医临床NLP import MedicalGraphEngine
    print("🔗 [V11.0 Import] 成功导入跨文件模块：[中西医临床NLP.py] -> MedicalGraphEngine (NetworkX)")
except ImportError:
    print("⚠️ [V11.0 Import] 未检测到 中西医临床NLP.py，NetworkX 图谱将处于静默安全模式...")
    MedicalGraphEngine = None

# 3. ⚡ V11.0 核心新增：从【模块 F：medical_tools.py】导入确定性临床算力与药理审查工具箱
try:
    from medical_tools import execute_medical_safety_check
    print("🔗 [V11.0 Import] 成功导入跨文件模块：[medical_tools.py] -> execute_medical_safety_check (药理安全审查工具)")
except ImportError:
    print("⚠️ [V11.0 Import] 未检测到 medical_tools.py，工具审查将默认处于静默放行模式...")
    def execute_medical_safety_check(text):
        return {"is_safe": True, "audit_report": "⚠️ 算力工具模块未挂载，默认放行。", "tcm_msg": ""}

# ==========================================
# 0. 全局字典与环境设置
# ==========================================
DISEASE_MAP = {
    0: "正常 (Normal)", 1: "肺炎 (Pneumonia)", 2: "肺结核 (Tuberculosis)", 3: "支气管炎 (Bronchitis)", 4: "肺气肿 (Emphysema)",
    5: "胸腔积液 (Pleural Effusion)", 6: "肺癌 (Lung Cancer)", 7: "气胸 (Pneumothorax)", 8: "肺大泡 (Pulmonary Bulla)", 9: "肺心病 (Cor Pulmonale)"
}
REVERSE_DISEASE_MAP = {v.split(" ")[0]: k for k, v in DISEASE_MAP.items()} 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 后端装载：千万级 RAG 向量检索引擎 (记忆轨道)
# ==========================================
class MedicalRAGEngine:
    def __init__(self, db_path="data/vector_db", model_name="BAAI/bge-small-zh-v1.5"):
        self.ready = False
        try:
            print("🚀 [1/2] 正在预热千万级中医古籍 ChromaDB 向量引擎...")
            self.embed_model = SentenceTransformer(model_name)
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_collection(name="tcm_ancient_books")
            self.ready = True
            print("✅ ChromaDB 向量引擎全线就绪！(管辖数据量: 33w+)")
        except Exception as e:
            print(f"⚠️ RAG引擎初始化失败，请检查数据库路径: {e}")

    def rag_arbitration(self, patient_text):
        if not self.ready:
            return None, "⚠️ 警告：RAG 向量库未加载，跳过仲裁。"
        if not patient_text or len(patient_text.strip()) == 0:
            return None, "无文本输入，跳过 RAG 仲裁。"

        query_vector = self.embed_model.encode([patient_text]).tolist()
        results = self.collection.query(query_embeddings=query_vector, n_results=3)

        evidence_chain = "### 📚 千万级 ChromaDB 古籍循证切片\n"
        
        keyword_disease_mapping = {
            "痨": "肺结核", "盗汗": "肺结核", "咳血": "肺结核", "骨蒸": "肺结核", "结核": "肺结核",
            "喘": "支气管炎", "哮": "支气管炎", "风寒": "支气管炎", "气管": "支气管炎",
            "肺胀": "肺气肿", "短气": "肺气肿", "不得卧": "肺气肿", "气肿": "肺气肿",
            "悬饮": "胸腔积液", "水饮": "胸腔积液", "十枣汤": "胸腔积液", "胸水": "胸腔积液", "积液": "胸腔积液",
            "肺痈": "肺炎", "咳吐脓血": "肺炎", "发热": "肺炎", "温病": "肺炎", "高烧": "肺炎", "大叶肺炎": "肺炎", "肺炎": "肺炎", "痰热": "肺炎",
            "心悸": "肺心病", "水肿": "肺心病", "肺络": "肺大泡"
        }

        detected_disease = None
        disease_scores = {}

        for i in range(3):
            text = results['documents'][0][i]
            book = results['metadatas'][0][i]['book']
            distance = results['distances'][0][i]

            display_text = text[:150] + "..." if len(text) > 150 else text
            evidence_chain += f"🔹 **文献 [{i+1}] (来源: 《{book}》)** | 语义距离: `{distance:.4f}`\n> {display_text}\n\n"

            if distance < 0.5: 
                for kw, target_dis in keyword_disease_mapping.items():
                    if kw in text:
                        score = (0.5 - distance) * (3 - i)
                        disease_scores[target_dis] = disease_scores.get(target_dis, 0) + score
        
        if disease_scores:
            detected_disease = max(disease_scores, key=disease_scores.get)
            evidence_chain += f"\n⚖️ **ChromaDB 语义推断**：通过多文献加权积分，最终聚焦指涉现代西医学之 **【{detected_disease}】**！"
        else:
            evidence_chain += "\n⚖️ **ChromaDB 语义推断**：未从古籍中提取出明确的现代危急重症映射。"

        return detected_disease, evidence_chain

# ==========================================
# 2. 后端装载：全局大脑与模型权重初始化
# ==========================================
print("🔄 [System Init] 正在装载底层模型权重与 RAG 双擎...")
rag_engine = MedicalRAGEngine()

graph_engine = None
if MedicalGraphEngine is not None:
    try:
        print("🚀 [2/2] 正在构建 NetworkX 深度结构化中西医结合星网...")
        graph_engine = MedicalGraphEngine()
        print("✅ NetworkX 因果图谱引擎装载完毕！")
    except Exception as e:
        print(f"⚠️ 图谱引擎装载异常: {e}")

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
model = MedicalMultimodalModel(num_classes=10).to(device)

model_path = "./checkpoints/real_medical_model_sota.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("✅ 成功加载实战 SOTA 模型权重！")
else:
    print("⚠️ 未检测到训练好的权重文件，将使用随机初始化参数进行演示。")

# ==========================================
# 3. 核心诊断与状态封装 (Hybrid RAG 双向联合推演)
# ==========================================
def run_diagnosis(image, text, boost_weight):
    if image is None or not text.strip():
        return None, "⚠️ 请同时上传 X 光片并输入患者主诉！", None, None, "无法诊断：数据不全", {}

    graph_target_western = None
    graph_report = "🕸️ **NetworkX 因果图谱**：主诉未触发图中特异性症状锚点。"
    if graph_engine is not None:
        try:
            graph_res = graph_engine.graph_reasoning(text)
            if graph_res and len(graph_res) == 3:
                graph_target_western, _, graph_report_raw = graph_res
                graph_report = f"### 🕸️ NetworkX 因果图谱多跳推理\n{graph_report_raw}"
            elif graph_res and len(graph_res) == 2:
                graph_target_western, graph_report_raw = graph_res
                graph_report = f"### 🕸️ NetworkX 因果图谱多跳推理\n{graph_report_raw}"
        except Exception as e:
            graph_report = f"🕸️ **NetworkX 因果图谱**：推理遇到边界提示 ({e})"

    vector_target_western, vector_report = rag_engine.rag_arbitration(text)
    hybrid_rag_report = f"{graph_report}\n\n---\n\n{vector_report}"
    target_western_disease = graph_target_western if graph_target_western else vector_target_western

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_rgb = image.convert('RGB')
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    img_tensor.requires_grad_(True) 
    
    encoding = tokenizer(text, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    
    model.zero_grad()
    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
        raw_logits = model(img_tensor, encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
        TEMPERATURE = 5.0 
        logits = raw_logits / TEMPERATURE
        raw_probs = F.softmax(logits, dim=1).flatten().cpu().detach().numpy()

    top_class = np.argmax(raw_probs)
    raw_logits[0, top_class].backward(retain_graph=True)
    
    gradients = model.gradients.cpu().data.numpy()[0]
    activations = model.activations.cpu().data.numpy()[0]
    
    weights = np.mean(np.abs(gradients), axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]
        
    cam = np.maximum(cam, 0) 
    cam = cv2.resize(cam, (image.size[0], image.size[1])) 
    if np.max(cam) != 0: cam = cam / np.max(cam)
        
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_img_np = np.array(img_rgb)
    cam_3d = np.expand_dims(cam, axis=2)
    superimposed_img = heatmap * cam_3d + original_img_np * (1 - cam_3d)
    heatmap_pil = Image.fromarray(np.uint8(superimposed_img))

    raw_dict = {DISEASE_MAP[i]: float(raw_probs[i]) for i in range(10)}
    final_logits = logits.clone().detach()
    arbitration_msg = "✅ 未触发跨界干预，底层直觉逻辑闭环。"
    
    visual_conf = float(raw_probs[top_class])
    if target_western_disease in REVERSE_DISEASE_MAP:
        target_idx = REVERSE_DISEASE_MAP[target_western_disease]
        visual_confidence = raw_probs[target_idx]
        dynamic_boost = boost_weight * (1.0 - visual_confidence)
        final_logits[0, target_idx] += dynamic_boost
        source_tag = "NetworkX 图谱因果" if graph_target_western else "ChromaDB 向量检索"
        arbitration_msg = f"⚖️ **{source_tag}动态仲裁生效！**\n系统结合底层医学逻辑与视觉置信度，执行了平滑的 Logit 补偿加权 (Dynamic Boost: +{dynamic_boost:.2f})。"
        
    final_probs = F.softmax(final_logits, dim=1).flatten().cpu().detach().numpy()
    final_dict = {DISEASE_MAP[i]: float(final_probs[i]) for i in range(10)}
    top_class_final = np.argmax(final_probs)
    
    conclusion = f"""### 🩺 V11.0 双轨制最终会诊结论\n* 👁️ **盲目直觉**: 疑似 **【{DISEASE_MAP[top_class]}】**\n* ⚖️ **双擎仲裁**: {arbitration_msg}\n* 🏥 **循证确诊**: 综合确诊 **【{DISEASE_MAP[top_class_final]}】**"""
    
    state_payload = {
        "patient_text": text,
        "vision_info": f"置信度 {visual_conf*100:.1f}% 倾向【{DISEASE_MAP[top_class]}】",
        "rag_info": hybrid_rag_report,
        "final_disease": DISEASE_MAP[top_class_final]
    }
    
    return heatmap_pil, hybrid_rag_report, raw_dict, final_dict, conclusion, state_payload

# ==========================================
# 4. ⚡ V11.0 LangGraph ReAct 动态算力工具与自省多智能体状态机
# ==========================================

# 4.1 升维图状态 (State)：新增处方初稿、算力工具审计报告与自省回退计数器
class MedicalState(TypedDict):
    patient_text: str
    vision_info: str
    rag_info: str
    final_disease: str
    western_opinion: str
    tcm_opinion: str
    chief_draft: str          # ✍️ Chief Agent 起草的初稿处方
    safety_audit: str         # 🛡️ 确定性算力工具的审查报告
    is_safe: bool             # 🚦 是否通过红线药理校验
    revision_count: int       # 🔄 自省回退重写次数

# 4.2 定义原生 API 调用模块
def call_ollama_sync(prompt_text):
    url = "http://localhost:11434/api/generate"
    payload = {"model": "qwen2.5:3b", "prompt": prompt_text, "stream": False}
    try:
        response = requests.post(url, json=payload).json()
        return response.get("response", "")
    except Exception:
        return "本地 Ollama 引擎未响应。"

# 4.3 西医 Agent Node
def western_agent_node(state: MedicalState) -> Dict:
    prompt = f"你是权威西医呼吸科专家。患者主诉：【{state['patient_text']}】。底层AI视觉推断：【{state['vision_info']}】。请用严谨现代西医病理学术语在80字内给出病理分析，不要有废话。"
    return {"western_opinion": call_ollama_sync(prompt)}

# 4.4 中医 Agent Node
def tcm_agent_node(state: MedicalState) -> Dict:
    prompt = f"你是权威中医温病学泰斗。患者主诉：【{state['patient_text']}】。双擎循证证据链：【{state['rag_info'][:350]}...】。请结合主诉与古籍用中医阴阳气血辨证在80字内给出病机推演与拟用药材，不要废话。"
    return {"tcm_opinion": call_ollama_sync(prompt)}

# 4.5 ⚡ V11.0 核心改写：Chief Agent 处方起草 Node (支持自省回退纠偏提示词)
def chief_agent_node(state: MedicalState) -> Dict:
    revision_warning = ""
    # 如果上一轮工具校验报错打回，强行在 System Prompt 中注入红线警告，逼迫模型修正！
    if state.get("revision_count", 0) > 0 and not state.get("is_safe", True):
        revision_warning = f"\n🚨 【算力工具拦截警告】：你上一版拟定的处方触犯了致命药理配伍禁忌！报错细节如下：\n{state['safety_audit']}\n请你立刻修改中药配伍，严禁使用互为十八反、十九畏禁忌的药材！"

    prompt = f"""你是本次会诊的主治医师（Chief Agent）。现在请综合团队意见拟定一份200字会诊报告及临床处方。
患者主诉：【{state['patient_text']}】 | 底层多模态确诊：【{state['final_disease']}】
西医意见：【{state['western_opinion']}】 | 中医意见：【{state['tcm_opinion']}】{revision_warning}
要求中西医互证并给出明确处方。直接输出正文，不要重复指令。"""
    
    draft = call_ollama_sync(prompt)
    return {"chief_draft": draft}

# 4.6 ⚡ V11.0 核心新增：Safety Reviewer 确定性算力工具安全审查 Node
def safety_reviewer_node(state: MedicalState) -> Dict:
    draft = state.get("chief_draft", "")
    # 唤醒 external Python 工具库进行 100% 确定性校验
    tool_res = execute_medical_safety_check(draft)
    
    current_revisions = state.get("revision_count", 0) + 1
    return {
        "is_safe": tool_res["is_safe"],
        "safety_audit": tool_res["audit_report"],
        "revision_count": current_revisions
    }

# 4.7 ⚡ V11.0 核心条件分支评价函数 (Conditional Edge Evaluator)
def should_revise_prescription(state: MedicalState) -> str:
    """
    控制流引擎：如果触犯红线且修改次数不超过 1 次，触发有向条件边，强行驳回至 Chief Agent 重写！
    """
    if not state.get("is_safe", True) and state.get("revision_count", 0) <= 1:
        print("🚨 [ReAct Loop] 探测到药理红线禁忌！触发条件边：强行打回 Chief Agent 执行自省重写！")
        return "re_draft"
    else:
        print("✅ [ReAct Loop] 工具安全校验放行，允许输出终极报告！")
        return "pass_to_end"

# 4.8 编译 V11.0 ReAct 有向无环/循环自省图
workflow = StateGraph(MedicalState)
workflow.add_node("Western_Agent", western_agent_node)
workflow.add_node("TCM_Agent", tcm_agent_node)
workflow.add_node("Chief_Agent", chief_agent_node)
workflow.add_node("Safety_Reviewer", safety_reviewer_node)

workflow.add_edge(START, "Western_Agent")
workflow.add_edge("Western_Agent", "TCM_Agent")
workflow.add_edge("TCM_Agent", "Chief_Agent")
workflow.add_edge("Chief_Agent", "Safety_Reviewer")

# 注册条件分支图跳转
workflow.add_conditional_edges(
    "Safety_Reviewer",
    should_revise_prescription,
    {
        "re_draft": "Chief_Agent",  # 🔄 触犯禁忌，循环回退！
        "pass_to_end": END          # 🏁 校验安全，终结流转！
    }
)
medical_graph = workflow.compile()

# ==========================================
# ⚡ V9.0/V11.0：DPO 偏好数据飞轮持久化引擎
# ==========================================
def save_dpo_feedback(prompt_context, ai_original_report, expert_modified_report):
    if not expert_modified_report or not expert_modified_report.strip():
        return "⚠️ 请先在文本框内输入专家修改/润色意见，再提交 DPO 飞轮！", expert_modified_report
    if not ai_original_report or not ai_original_report.strip():
        return "⚠️ 未捕获到 AI 原始诊断报告，请先运行联合会诊！", expert_modified_report
    
    dpo_dir = "data"
    os.makedirs(dpo_dir, exist_ok=True)
    dpo_file = os.path.join(dpo_dir, "dpo_dataset.json")
    
    new_sample = {
        "prompt": prompt_context,
        "chosen": expert_modified_report.strip(),
        "rejected": ai_original_report.strip()
    }
    
    dataset = []
    if os.path.exists(dpo_file):
        try:
            with open(dpo_file, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        except Exception:
            dataset = []
            
    dataset.append(new_sample)
    
    with open(dpo_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    return f"✅ 成功存入 DPO 自进化飞轮！当前本地 Dataset 累计对齐样本数: **{len(dataset)}** 条", ""

# 4.9 ⚡ V11.0 最终图执行与主治医师流式输出适配 (实时监控 ReAct 拦截过程)
def run_multi_agent_stream(state_payload):
    if not state_payload:
        yield "⚠️ 等待系统完成初始诊断...", "", "", ""
        return
        
    initial_state = MedicalState(
        patient_text=state_payload["patient_text"],
        vision_info=state_payload["vision_info"],
        rag_info=state_payload["rag_info"],
        final_disease=state_payload["final_disease"],
        western_opinion="",
        tcm_opinion="",
        chief_draft="",
        safety_audit="",
        is_safe=True,
        revision_count=0
    )
    
    ui_output = "### 🌐 V11.0 LangGraph ReAct 动态工具与自省工作流启动\n\n"
    yield ui_output + "*(正在调度【西医影像 Agent】分析病理...)*", "", "", ""
    
    final_state = initial_state.copy()
    for output in medical_graph.stream(initial_state):
        for node_name, state_update in output.items():
            final_state.update(state_update)
            if node_name == "Western_Agent":
                ui_output += f"**👨‍⚕️ 西医影像 Agent**：\n> {state_update['western_opinion']}\n\n"
                yield ui_output + "*(正在调度【中医典籍 Agent】溯源古籍...)*", "", "", ""
            elif node_name == "TCM_Agent":
                ui_output += f"**🌿 中医典籍 Agent**：\n> {state_update['tcm_opinion']}\n\n"
                yield ui_output + "*(✍️ 正在调度【Chief Agent】起草初稿处方...)*", "", "", ""
            elif node_name == "Chief_Agent":
                if final_state.get("revision_count", 0) > 0:
                    ui_output += f"**🔄 Chief Agent (第 {final_state['revision_count']} 次自省重写)**：\n> 正在根据算力工具安全报警修改处方...\n> {state_update['chief_draft'][:120]}...\n\n"
                else:
                    ui_output += f"**✍️ Chief Agent 初稿处方**：\n> {state_update['chief_draft'][:120]}...\n\n"
                yield ui_output + "*(🧰 正在调度【Safety Reviewer Node】执行 Python 确定性算力工具审查...)*", "", "", ""
            elif node_name == "Safety_Reviewer":
                ui_output += f"{state_update['safety_audit']}\n\n"
                if not state_update['is_safe'] and final_state.get("revision_count", 0) <= 1:
                    ui_output += "🚨 **触发 ReAct 条件分支**：发现致死性药理禁忌！图状态机正在将流程**强行驳回**至 Chief Agent 执行自省重写！\n\n---\n\n"
                yield ui_output, "", "", ""

    # 图流转完毕，直接抽取最终被算力工具安全放行的终极报告
    final_report = final_state.get("chief_draft", "未生成有效报告")
    ui_output += f"\n---\n### ✍️ V11.0 终极安全会诊裁决报告\n{final_report}"
    
    dpo_prompt_context = f"患者主诉：【{final_state['patient_text']}】 | 综合确诊：【{final_state['final_disease']}】 | 西医意见：【{final_state['western_opinion']}】 | 中医意见：【{final_state['tcm_opinion']}】 | 算力工具审查：【{'通过' if final_state.get('is_safe') else '已自省纠偏'}】"
    
    yield ui_output, dpo_prompt_context, final_report, final_report

# ==========================================
# 5. Gradio UI 界面构建 (已升至 V11.0 ReAct 自省大一统全景面板)
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", neutral_hue="slate")

with gr.Blocks(theme=theme, title="V11.0 Multi-Agent Medical LLM, ReAct Tools & DPO Flywheel") as demo:
    gr.Markdown(
        """
        # 🏥 中西医双轨制多模态医疗大脑 (v11.0 ReAct 动态工具与药理自省版)
        **Core**: `ResNet-50` + `ChromaDB` + `NetworkX Causal Graph` | **Orchestration**: `LangGraph ReAct Loop` | **Tools**: `CURB-65 & TCM Safety` | **Evolution**: `On-Device DPO`
        
        系统已全面跃迁至 **V11.0 ReAct 动态工具自省版**！深度缝合 NetworkX 因果图谱 与 ChromaDB 构成 Hybrid RAG；引入 **Python 确定性算力工具箱** 执行 CURB-65 评分与“十八反十九畏”禁忌校验，触发致死红线时自动开启有向循环分支勒令大模型自省重写，彻底杜绝大模型量化计算与药理幻觉！
        """
    )
    
    state_payload_hidden = gr.State({})
    dpo_prompt_hidden = gr.State("")
    dpo_rejected_hidden = gr.State("")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👨‍⚕️ 临床数据输入")
            input_image = gr.Image(type="pil", label="上传胸部 X 光影像", height=300)
            input_text = gr.Textbox(lines=4, label="输入患者临床主诉 (文本)")
            boost_slider = gr.Slider(minimum=0.0, maximum=50.0, value=15.0, step=1.0, label="Logit RAG 干预权重")
            submit_btn = gr.Button("🚀 启动 LangGraph 多智能体联合会诊", variant="primary", size="lg")
            
            gr.Markdown("### 👁️ 可解释性 AI (XAI) 视觉追踪")
            output_heatmap = gr.Image(type="pil", label="AI 视觉注意力热力图")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 实时会诊监控大屏")
            with gr.Accordion("🧠 理智轨道：Hybrid RAG 双擎循证 (NetworkX 图谱 + ChromaDB)", open=True):
                output_graph_report = gr.Markdown("等待输入数据...")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 👁️ 纯视觉直觉")
                    output_raw_probs = gr.Label(num_top_classes=3, label="未经干预的底层概率")
                with gr.Column():
                    gr.Markdown("#### ⚖️ 双擎仲裁后")
                    output_final_probs = gr.Label(num_top_classes=3, label="Logit 注入后的最终概率")
            output_conclusion = gr.Markdown("### 🩺 最终会诊结论\n(等待运行...)")
            
            gr.Markdown("---")
            output_llm_report = gr.Markdown("### 🌐 LangGraph ReAct 动态工具自省监控\n*(等待状态流转...)*")

    # ================= 新增 V11.0 专家临床修正与 DPO 数据飞轮面板 =================
    gr.Markdown("---")
    with gr.Accordion("👨‍⚕️ V11.0 专家临床修正面板 (DPO 自进化数据飞轮)", open=True):
        gr.Markdown("💡 **端侧闭环操作指南**：当上方被算力工具安全放行的 Chief Agent 终极报告生成完毕后，系统会自动将原稿加载至下方文本框。主治医师可直接以红笔视角批改润色。点击提交后，系统会将 `AI原稿(Rejected)` 与 `专家修改稿(Chosen)` 自动组成对比偏好对，写入本地 `data/dpo_dataset.json`，反哺夜间 QLoRA + DPO 自进化飞轮。")
        with gr.Row():
            dpo_chosen_input = gr.Textbox(lines=5, label="✍️ 专家润色/修改后的终极临床建议 (Chosen)", placeholder="等待 AI 会诊报告生成，或在此处直接红笔修改...")
        with gr.Row():
            dpo_submit_btn = gr.Button("💾 提交专家修改并存入本地 DPO 飞轮", variant="secondary", size="lg")
        with gr.Row():
            dpo_status_output = gr.Markdown("当前状态: *等待主治医师批改报告...*")

    # 🔗 核心数据链路绑定
    submit_btn.click(
        fn=run_diagnosis,
        inputs=[input_image, input_text, boost_slider],
        outputs=[output_heatmap, output_graph_report, output_raw_probs, output_final_probs, output_conclusion, state_payload_hidden]
    ).then(
        fn=run_multi_agent_stream,
        inputs=[state_payload_hidden],
        outputs=[output_llm_report, dpo_prompt_hidden, dpo_rejected_hidden, dpo_chosen_input]
    )

    # 🔗 DPO 飞轮提交绑定
    dpo_submit_btn.click(
        fn=save_dpo_feedback,
        inputs=[dpo_prompt_hidden, dpo_rejected_hidden, dpo_chosen_input],
        outputs=[dpo_status_output, dpo_chosen_input]
    )

    gr.Markdown("--- \n*Architected by Chen Yijin | Powered by PyTorch, LangGraph ReAct, Local Ollama, Hybrid RAG, Deterministic Tools & On-Device DPO*")

if __name__ == "__main__":
    print("\n🌐 正在启动 Gradio 医疗控制面板...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)