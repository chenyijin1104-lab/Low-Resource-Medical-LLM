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

# ================= 新增 V8.0 LangGraph 多智能体依赖 =================
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, START, END

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
# 1. 后端装载：千万级 RAG 向量检索引擎 
# ==========================================
class MedicalRAGEngine:
    def __init__(self, db_path="data/vector_db", model_name="BAAI/bge-small-zh-v1.5"):
        self.ready = False
        try:
            print("🚀 正在预热千万级中医古籍向量引擎...")
            self.embed_model = SentenceTransformer(model_name)
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_collection(name="tcm_ancient_books")
            self.ready = True
            print("✅ RAG 向量引擎全线就绪！(管辖数据量: 33w+)")
        except Exception as e:
            print(f"⚠️ RAG引擎初始化失败，请检查数据库路径: {e}")

    def rag_arbitration(self, patient_text):
        if not self.ready:
            return None, "⚠️ 警告：RAG 向量库未加载，跳过仲裁。"
        if not patient_text or len(patient_text.strip()) == 0:
            return None, "无文本输入，跳过 RAG 仲裁。"

        query_vector = self.embed_model.encode([patient_text]).tolist()
        results = self.collection.query(query_embeddings=query_vector, n_results=3)

        evidence_chain = "### 🧠 千万级 RAG 古籍循证链条已触发\n"
        
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
            evidence_chain += f"\n⚖️ **RAG 语义推断**：通过多文献特征加权积分投票，最终聚焦指涉现代西医学之 **【{detected_disease}】**，触发动态特征融合机制！"
        else:
            evidence_chain += "\n⚖️ **RAG 语义推断**：未从古籍中提取出明确的现代危急重症映射，维持神经网络直觉诊断。"

        return detected_disease, evidence_chain

# ==========================================
# 2. 后端装载：v5.0 多模态网络
# ==========================================
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

print("🔄 正在装载底层模型权重与RAG库...")
rag_engine = MedicalRAGEngine()
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')
model = MedicalMultimodalModel(num_classes=10).to(device)

model_path = "./checkpoints/real_medical_model_sota.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("✅ V5.0 模型权重加载成功！")
else:
    print("⚠️ 未检测到训练好的权重文件，将使用随机初始化参数进行演示。")

# ==========================================
# 3. 核心诊断与状态封装 (Phase 1)
# ==========================================
def run_diagnosis(image, text, boost_weight):
    if image is None or not text.strip():
        return None, "⚠️ 请同时上传 X 光片并输入患者主诉！", None, None, "无法诊断：数据不全", {}

    target_western_disease, rag_report = rag_engine.rag_arbitration(text)
    
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
        arbitration_msg = f"⚖️ **动态自适应 RAG 仲裁生效！**\n系统综合评估了视觉网络的置信度。执行了平滑的 Logit 补偿加权 (Dynamic Boost: +{dynamic_boost:.2f})。"
        
    final_probs = F.softmax(final_logits, dim=1).flatten().cpu().detach().numpy()
    final_dict = {DISEASE_MAP[i]: float(final_probs[i]) for i in range(10)}
    top_class_final = np.argmax(final_probs)
    
    conclusion = f"""### 🩺 双轨制最终会诊结论\n* 👁️ **盲目直觉**: 疑似 **【{DISEASE_MAP[top_class]}】**\n* ⚖️ **仲裁状态**: {arbitration_msg}\n* 🏥 **循证确诊**: 综合确诊 **【{DISEASE_MAP[top_class_final]}】**"""
    
    # 🎯 V8.0 核心：不再组装单一 Prompt，而是将全局状态打包成字典，准备喂给 LangGraph
    state_payload = {
        "patient_text": text,
        "vision_info": f"置信度 {visual_conf*100:.1f}% 倾向【{DISEASE_MAP[top_class]}】",
        "rag_info": rag_report if target_western_disease else "未检出特定重症",
        "final_disease": DISEASE_MAP[top_class_final]
    }
    
    return heatmap_pil, rag_report, raw_dict, final_dict, conclusion, state_payload

# ==========================================
# 4. V8.0 LangGraph 多智能体编排引擎 (Phase 2)
# ==========================================

# 4.1 定义图状态 (State)
class MedicalState(TypedDict):
    patient_text: str
    vision_info: str
    rag_info: str
    final_disease: str
    western_opinion: str
    tcm_opinion: str

# 4.2 定义原生 API 调用模块 (Zero-bloat)
def call_ollama_sync(prompt_text):
    url = "http://localhost:11434/api/generate"
    payload = {"model": "qwen2.5:3b", "prompt": prompt_text, "stream": False}
    try:
        response = requests.post(url, json=payload).json()
        return response.get("response", "")
    except Exception:
        return "本地 Ollama 引擎未响应。"

# 4.3 定义西医 Agent (Node A)
def western_agent_node(state: MedicalState) -> Dict:
    prompt = f"你是权威的西医呼吸科专家。患者主诉：【{state['patient_text']}】。底层AI视觉网络对X光片的推断为：【{state['vision_info']}】。请用极度专业的现代西医病理学术语，在80字以内给出纯西医视角的临床病理分析。不要有任何废话。"
    opinion = call_ollama_sync(prompt)
    return {"western_opinion": opinion}

# 4.4 定义中医 Agent (Node B)
def tcm_agent_node(state: MedicalState) -> Dict:
    prompt = f"你是权威的中医温病学泰斗。患者主诉：【{state['patient_text']}】。我们从33万卷古籍中为你提取了相关医案证据：【{state['rag_info'][:300]}...】。请结合主诉与古籍，用专业的中医辨证术语（如阴阳气血、脏腑经络），在80字以内给出纯中医视角的病机推演。不要有任何废话。"
    opinion = call_ollama_sync(prompt)
    return {"tcm_opinion": opinion}

# 4.5 编译 LangGraph 状态机
workflow = StateGraph(MedicalState)
workflow.add_node("Western_Agent", western_agent_node)
workflow.add_node("TCM_Agent", tcm_agent_node)

# 强制串行执行，榨干 RTX 4050 显存的同时防止 OOM
workflow.add_edge(START, "Western_Agent")
workflow.add_edge("Western_Agent", "TCM_Agent")
workflow.add_edge("TCM_Agent", END)
medical_graph = workflow.compile()

# 4.6 最终图执行与主治医师(Chief Agent)流式输出
def run_multi_agent_stream(state_payload):
    if not state_payload:
        yield "⚠️ 等待系统完成初始诊断..."
        return
        
    initial_state = MedicalState(
        patient_text=state_payload["patient_text"],
        vision_info=state_payload["vision_info"],
        rag_info=state_payload["rag_info"],
        final_disease=state_payload["final_disease"],
        western_opinion="",
        tcm_opinion=""
    )
    
    ui_output = "### 🌐 LangGraph 多智能体工作流启动\n\n"
    yield ui_output + "*(正在调度【西医影像 Agent】分析病理...)*"
    
    # 执行 LangGraph，并捕获每个节点的增量状态更新UI
    final_state = initial_state.copy()
    for output in medical_graph.stream(initial_state):
        for node_name, state_update in output.items():
            final_state.update(state_update)
            if node_name == "Western_Agent":
                ui_output += f"**👨‍⚕️ 西医影像 Agent**：\n> {state_update['western_opinion']}\n\n"
                yield ui_output + "*(正在调度【中医典籍 Agent】溯源古籍...)*"
            elif node_name == "TCM_Agent":
                ui_output += f"**🌿 中医典籍 Agent**：\n> {state_update['tcm_opinion']}\n\n"
                yield ui_output + "*(⚖️ 状态图流转完毕！Chief Agent 正在执行联合裁决与流式报告生成...)*"

    # 图执行完毕，唤醒 Chief Agent 进行打字机输出
    chief_prompt = f"""你是本次会诊的主治医师（Chief Agent）。现在你需要融合团队给出的诊断。
患者主诉：【{final_state['patient_text']}】
底层多模态网络最终确诊：【{final_state['final_disease']}】
西医专家的意见：【{final_state['western_opinion']}】
中医专家的意见：【{final_state['tcm_opinion']}】

请你起草一份200字左右的《V8.0 多智能体联合会诊最终报告》。要求逻辑自洽，中西医互证，并给出最终干预建议。直接输出报告正文，不要重复指令。"""

    ui_output += "\n---\n### ✍️ 主治医师 (Chief Agent) 最终裁决报告\n"
    
    # 纯原生流式拉取
    url = "http://localhost:11434/api/generate"
    payload = {"model": "qwen2.5:3b", "prompt": chief_prompt, "stream": True}
    
    try:
        response = requests.post(url, json=payload, stream=True)
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                ui_output += chunk.get("response", "")
                yield ui_output 
    except requests.exceptions.ConnectionError:
        yield ui_output + "\n\n❌ 本地大模型服务未启动。"

# ==========================================
# 5. Gradio UI 界面构建
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", neutral_hue="slate")

with gr.Blocks(theme=theme, title="V8.0 Multi-Agent Medical LLM") as demo:
    gr.Markdown(
        """
        # 🏥 中西医双轨制多模态医疗大脑 (v8.0 Multi-Agent 架构版)
        **Core**: `ResNet-50` + `ChromaDB` | **Orchestration**: `LangGraph` | **Agents**: `Qwen2.5 (Local)`
        
        系统已全面跃迁至**多智能体协作(Multi-Agent)**架构！在 LangGraph 有向无环图的精准调度下，【西医 Agent】与【中医 Agent】将首先进行独立推演，最后由【Chief Agent】进行跨模态流式仲裁。
        """
    )
    
    state_payload_hidden = gr.State({})
    
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
            with gr.Accordion("🧠 理智轨道：千万级 RAG 循证报告", open=True):
                output_graph_report = gr.Markdown("等待输入数据...")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 👁️ 纯视觉直觉")
                    output_raw_probs = gr.Label(num_top_classes=3, label="未经干预的底层概率")
                with gr.Column():
                    gr.Markdown("#### ⚖️ RAG 仲裁后")
                    output_final_probs = gr.Label(num_top_classes=3, label="Logit 注入后的最终概率")
            output_conclusion = gr.Markdown("### 🩺 最终会诊结论\n(等待运行...)")
            
            gr.Markdown("---")
            output_llm_report = gr.Markdown("### 🌐 LangGraph 多智能体工作流监控\n*(等待状态流转...)*")

    submit_btn.click(
        fn=run_diagnosis,
        inputs=[input_image, input_text, boost_slider],
        outputs=[output_heatmap, output_graph_report, output_raw_probs, output_final_probs, output_conclusion, state_payload_hidden]
    ).then(
        fn=run_multi_agent_stream,
        inputs=[state_payload_hidden],
        outputs=[output_llm_report]
    )

    gr.Markdown("--- \n*Architected by Chen Yijin | Powered by PyTorch, LangGraph & Local Ollama*")

if __name__ == "__main__":
    print("\n🌐 正在启动 Gradio 医疗控制面板...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)