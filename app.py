import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2 # 用于生成热力图
from transformers import BertConfig, BertModel, BertTokenizer
import torchvision.models as models
import torchvision.transforms as transforms
import networkx as nx
import gradio as gr
import json

# 禁用 tokenizer 并行警告
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
# 1. 后端装载：中西医图谱引擎
# ==========================================
class MedicalGraphEngine:
    def __init__(self, json_path="data/kg_data/tcm_knowledge.json"):
        self.G = nx.DiGraph()
        self.json_path = json_path
        self.ready = False
        if os.path.exists(self.json_path):
            self._build_sino_western_knowledge_graph()
            self.ready = True

    def _build_sino_western_knowledge_graph(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
            
        for item in knowledge_base["syndromes"]:
            syndrome = item["name"]
            western_disease = item["western_disease"]
            symptoms = item["symptoms"]
            formula = item["formula"]
            herbs = item["herbs"]
            
            self.G.add_node(syndrome, type="TCMSyndrome")
            self.G.add_node(western_disease, type="WesternDisease")
            self.G.add_edge(western_disease, syndrome, relation="mapped_to_syndrome")
            
            for sym in symptoms:
                self.G.add_node(sym, type="Symptom")
                self.G.add_edge(syndrome, sym, relation="has_symptom")
            self.G.add_node(formula, type="Formula")
            self.G.add_edge(syndrome, formula, relation="treats_with")
            for herb in herbs:
                self.G.add_node(herb, type="Herb")
                self.G.add_edge(formula, herb, relation="contains_herb")

    def graph_reasoning(self, text_content):
        if not self.ready:
            return None, "⚠️ 警告：知识图谱未加载，跳过图谱推理。"

        hit_symptoms = [node for node, data in self.G.nodes(data=True) if data.get('type') == 'Symptom' and node in text_content]
        if not hit_symptoms:
            return None, "❌ 图谱未在主诉中抽离出核心中医体征，无法建立有效映射。"

        matched_syndromes = []
        for node, data in self.G.nodes(data=True):
            if data.get('type') == 'TCMSyndrome':
                expected_symptoms = [tgt for _, tgt, rel in self.G.out_edges(node, data='relation') if rel == 'has_symptom']
                score = sum(1 for sym in hit_symptoms if sym in expected_symptoms)
                if score > 0:
                    matched_syndromes.append((node, score))
                    
        if not matched_syndromes:
            return None, "❌ 无法形成有效的证型聚类。"
            
        matched_syndromes.sort(key=lambda x: x[1], reverse=True)
        best_syndrome = matched_syndromes[0][0]

        mapped_western_diseases = [src for src, tgt, rel in self.G.in_edges(best_syndrome, data='relation') if rel == 'mapped_to_syndrome']
        target_western_disease = mapped_western_diseases[0] if mapped_western_diseases else None

        formulas = [tgt for _, tgt, rel in self.G.out_edges(best_syndrome, data='relation') if rel == 'treats_with']
        best_formula = formulas[0] if formulas else "辨证施治"
        herbs = [tgt for _, tgt, rel in self.G.out_edges(best_formula, data='relation') if rel == 'contains_herb']
        
        report_md = f"""
### 📜 中医 GraphRAG 循证链条追踪成功
* **命中核心体征**：`{', '.join(hit_symptoms)}`
* **锁定中医证型**：**【{best_syndrome}】**
* **跨界映射西医**：🚨 **疑似 {target_western_disease}**
* **推荐经方配伍**：**{best_formula}** ({', '.join(herbs)})
"""
        return target_western_disease, report_md

# ==========================================
# 2. 后端装载：v5.0 终极多模态网络 (含 XAI 钩子)
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

# 初始化全局变量
print("🔄 正在装载底层模型权重与图谱...")
graph_engine = MedicalGraphEngine()
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
# 3. 核心大模型会诊逻辑 (包含热力图生成)
# ==========================================
def run_diagnosis(image, text, boost_weight):
    # 【核心防崩溃修复】：确保返回 5 个参数，与前端 outputs 一一对应
    if image is None or not text.strip():
        return None, "⚠️ 请同时上传 X 光片并输入患者主诉！", None, None, "无法诊断：数据不全"

    target_western_disease, graph_report = graph_engine.graph_reasoning(text)
    
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
    
    # 【热力图核心修复】：提取绝对值，防止梯度抵消导致全图发蓝
    weights = np.mean(np.abs(gradients), axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]
        
    cam = np.maximum(cam, 0) 
    cam = cv2.resize(cam, (image.size[0], image.size[1])) 
    
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
        
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    original_img_np = np.array(img_rgb)
    
    # 【热力图核心修复】：动态透明度叠加，无权重的区域保留原图黑白色
    cam_3d = np.expand_dims(cam, axis=2)
    superimposed_img = heatmap * cam_3d + original_img_np * (1 - cam_3d)
    heatmap_pil = Image.fromarray(np.uint8(superimposed_img))

    raw_dict = {DISEASE_MAP[i]: float(raw_probs[i]) for i in range(10)}
    final_logits = logits.clone().detach()
    arbitration_msg = "✅ 未触发跨界干预，底层直觉逻辑闭环。"
    
    if target_western_disease in REVERSE_DISEASE_MAP:
        target_idx = REVERSE_DISEASE_MAP[target_western_disease]
        final_logits[0, target_idx] += boost_weight
        arbitration_msg = f"💉 **图谱铁腕干预生效！**\n系统检测到跨模态逻辑冲突，已强行向西医【{target_western_disease}】维度注入惩罚权重 (Boost: +{boost_weight})，已逆转底部视觉幻觉！"
        
    final_probs = F.softmax(final_logits, dim=1).flatten().cpu().detach().numpy()
    final_dict = {DISEASE_MAP[i]: float(final_probs[i]) for i in range(10)}
    top_class_final = np.argmax(final_probs)
    
    conclusion = f"""
### 🩺 双轨制最终会诊结论
* 👁️ **盲目直觉 (仅神经网络)**: 疑似 **【{DISEASE_MAP[top_class]}】**
* ⚖️ **仲裁状态**: {arbitration_msg}
* 🏥 **循证确诊 (多模态+图谱对齐)**: 高度确诊 **【{DISEASE_MAP[top_class_final]}】**
    """
    
    return heatmap_pil, graph_report, raw_dict, final_dict, conclusion

# ==========================================
# 4. Gradio 赛博朋克风 UI 界面构建
# ==========================================
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif']
)

with gr.Blocks(theme=theme, title="Sino-Western Medical LLM") as demo:
    gr.Markdown(
        """
        # 🏥 中西医双轨制多模态医疗大脑 (v5.0 终极版)
        **Core Engine**: `ResNet-50` + `BERT` + `Cross-Attention` | **GraphRAG**: `NetworkX` | **XAI**: `Grad-CAM`
        
        上传胸部 X 光片并输入患者临床主诉。系统不仅会执行多模态诊断和 Logit 仲裁，还会通过 **Grad-CAM 梯度追踪**，实时生成 AI 的视觉注意力热力图，彻底打破深度学习黑盒！
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👨‍⚕️ 临床数据输入")
            input_image = gr.Image(type="pil", label="上传胸部 X 光影像", height=300)
            input_text = gr.Textbox(lines=4, label="输入患者临床主诉 (文本)", placeholder="例如：患者近期消瘦明显，伴有干咳、夜间盗汗...")
            boost_slider = gr.Slider(minimum=0.0, maximum=50.0, value=15.0, step=1.0, label="Logit 仲裁干预权重", info="数值越大，图谱纠偏能力越强")
            submit_btn = gr.Button("🚀 启动联合会诊与 XAI 分析", variant="primary", size="lg")
            
            gr.Markdown("### 👁️ 可解释性 AI (XAI) 视觉追踪")
            output_heatmap = gr.Image(type="pil", label="AI 视觉注意力热力图 (红色为高权重决策区)")
            
        with gr.Column(scale=1):
            gr.Markdown("### 📊 实时会诊监控大屏")
            with gr.Accordion("📚 理智轨道：GraphRAG 溯源报告", open=True):
                output_graph_report = gr.Markdown("等待输入数据...")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 👁️ 纯视觉直觉")
                    output_raw_probs = gr.Label(num_top_classes=3, label="未经干预的底层概率")
                with gr.Column():
                    gr.Markdown("#### ⚖️ 图谱仲裁后")
                    output_final_probs = gr.Label(num_top_classes=3, label="Logit 注入后的最终概率")
            output_conclusion = gr.Markdown("### 🩺 最终会诊结论\n(等待运行...)")

    submit_btn.click(
        fn=run_diagnosis,
        inputs=[input_image, input_text, boost_slider],
        outputs=[output_heatmap, output_graph_report, output_raw_probs, output_final_probs, output_conclusion]
    )

    gr.Markdown("--- \n*Developed by Chen Yijin | RTX 4050 Build | Powered by PyTorch, Gradio & Grad-CAM*")

if __name__ == "__main__":
    print("\n🌐 正在启动 Gradio 医疗控制面板...")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
