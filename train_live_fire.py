import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel

# ==========================================
# 0. 跨文件调用：连接你的真实数据兵工厂
# ==========================================
try:
    from real_data_pipeline import KaggleXRayMultimodalDataset
except ImportError:
    raise ImportError("❌ 致命错误：找不到 real_data_pipeline.py！请确保它和当前文件在同一目录下。")

# 禁用 tokenizer 并行警告，保持控制台清爽
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. 核心视觉大脑 (保持 10 分类，无缝对接推理端)
# ==========================================
class MedicalMultimodalModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # CNN 视觉直觉提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((16, 16))
        )
        self.img_proj = nn.Linear(64, 256)
        
        # BERT 临床文本提取器 (参数与你前期保持绝对一致)
        self.bert = BertModel(BertConfig(
            vocab_size=21128, hidden_size=256, num_hidden_layers=4, 
            num_attention_heads=4, intermediate_size=1024
        ))
        
        # 跨模态注意力机制 (灵魂组件)
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True, dropout=0.3)
        
        # 最终分类器
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
# 2. 工业级实弹炼丹引擎 (防 OOM 优化版)
# ==========================================
def train_on_real_data(epochs=3, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 启动实弹炼丹引擎！")
    print(f"💻 当前接管设备: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    
    if device.type == 'cuda':
        print("⚡ 已开启 AMP (自动混合精度)，全功率保护 RTX 4050 显存！")

    # 1. 挂载真实数据
    print("📦 正在挂载 Kaggle 1.15GB 真实医疗影像库...")
    try:
        train_dataset = KaggleXRayMultimodalDataset(data_dir="./chest_xray/train")
        # batch_size=8 是 4050 的绝对安全区
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 实例化大脑并推入显卡
    model = MedicalMultimodalModel(num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    
    # 混合精度缩放器 (PyTorch 2.x 标准写法)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    print("\n" + "="*50)
    print(f"🚀 开始真实的梯度下降！目标 Epochs: {epochs}")
    print("="*50)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_preds = 0
        total_samples = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # 将张量弹药推入显存
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # 前向传播 (混合精度上下文)
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                logits = model(images, input_ids, attention_mask)
                
                # --- 你的绝版底层架构：MAXSUP 防作弊修复 ---
                ce_loss = F.cross_entropy(logits, labels)
                max_logits, _ = torch.max(logits, dim=1)
                # 平方惩罚项：死死锚定 0 点，防止 Loss 塌陷到负数
                maxsup_loss = ce_loss + 0.1 * torch.mean(max_logits ** 2) 

            # 反向传播与梯度缩放
            scaler.scale(maxsup_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计战况
            total_loss += maxsup_loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            # 每 50 个 Batch 汇报一次实时战况
            if batch_idx % 50 == 0:
                current_acc = 100 * correct_preds / total_samples
                print(f"⏳ Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | 实时 Loss: {maxsup_loss.item():.4f} | 批次准确率: {current_acc:.1f}%")

        # 结算当前 Epoch
        epoch_time = time.time() - start_time
        epoch_acc = 100 * correct_preds / total_samples
        print("-" * 50)
        print(f"✅ Epoch {epoch+1} 结束! 耗时: {epoch_time:.1f}s | 平均 Loss: {total_loss/len(train_loader):.4f} | 真实影像总准确率: {epoch_acc:.2f}%")
        print("-" * 50)

    # 3. 固化战斗成果
    save_path = "real_medical_model_sota.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n🏆 炼丹完美收官！模型已在真实世界开眼！")
    print(f"💾 实战权重已极其安全地保存至: {save_path}")

if __name__ == "__main__":
    # 为了验证架构连通性，先跑 2 个 Epoch
    train_on_real_data(epochs=2, batch_size=8)
