import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel
import torchvision.models as models # 引入视觉顶尖模型库

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
# 1. 核心视觉大脑 (v5.0 终极版：潜空间对齐架构)
# ==========================================
class MedicalMultimodalModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 引入预训练的 ResNet-50，截断全局池化和全连接层
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # 冻结前 80% 网络，防爆显存并加速收敛
        for param in self.cnn[:-1].parameters():
            param.requires_grad = False
            
        # 线性投影：将 ResNet 的 2048 维度压缩对齐至 BERT 的 256 维
        self.img_proj = nn.Linear(2048, 256)
        
        # BERT 临床文本提取器 
        self.bert = BertModel(BertConfig(
            vocab_size=21128, hidden_size=256, num_hidden_layers=4, 
            num_attention_heads=4, intermediate_size=1024
        ))
        
        # 跨模态注意力机制
        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True, dropout=0.3)
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 128), nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(128, num_classes)
        )
        
    # 【核心升级】：加入了 return_embeddings 参数，完美兼容以前的推理代码
    def forward(self, images, input_ids, attention_mask, return_embeddings=False):
        # 处理单通道输入
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
            
        img_f = self.cnn(images)
        img_seq = img_f.view(img_f.size(0), 2048, -1).permute(0, 2, 1)
        img_seq = F.relu(self.img_proj(img_seq)) # 形状: (Batch, 64, 256)
        
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        txt_pooler = bert_out.pooler_output # 形状: (Batch, 256)
        
        attn_out, _ = self.cross_attn(query=txt_pooler.unsqueeze(1), key=img_seq, value=img_seq)
        fused_features = torch.cat((attn_out.squeeze(1), txt_pooler), dim=1)
        
        logits = self.classifier(fused_features)
        
        # 如果是训练模式，不仅吐出分类结果，还要把图像和文本的“灵魂（特征向量）”单独抽出来供对比学习使用
        if return_embeddings:
            img_pooler = img_seq.mean(dim=1) # 把 64 个区块取平均，浓缩成 256 维图像向量
            return logits, img_pooler, txt_pooler
            
        return logits

# ==========================================
# 2. 工业级实弹炼丹引擎 (双擎驱动版)
# ==========================================
def train_on_real_data(epochs=3, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 启动实弹炼丹引擎 (V5.0 终极版：InfoNCE 潜空间强制对齐)！")
    print(f"💻 当前接管设备: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    
    if device.type == 'cuda':
        print("⚡ 已开启 AMP (自动混合精度)，全功率保护 RTX 4050 显存！")

    # 1. 挂载真实数据
    print("📦 正在挂载 Kaggle 1.15GB 真实医疗影像库...")
    try:
        train_dataset = KaggleXRayMultimodalDataset(data_dir="./data/chest_xray/train")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 2. 实例化大脑并推入显卡
    model = MedicalMultimodalModel(num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    print("\n" + "="*70)
    print(f"🚀 开始多模态【双引擎】真实梯度下降！目标 Epochs: {epochs}")
    print("="*70)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_preds = 0
        total_samples = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                # 提取出分类概率(logits) 以及 纯净的图像/文本特征向量
                logits, img_emb, txt_emb = model(images, input_ids, attention_mask, return_embeddings=True)
                
                # ==================================================
                # 【引擎 1】：MaxSup 传统分类交叉熵损失
                # ==================================================
                ce_loss = F.cross_entropy(logits, labels)
                max_logits, _ = torch.max(logits, dim=1)
                maxsup_loss = ce_loss + 0.1 * torch.mean(max_logits ** 2) 

                # ==================================================
                # 【引擎 2】：InfoNCE 对比学习损失 (潜空间强行对齐)
                # ==================================================
                # 1. 高维空间张量归一化，变成只看方向不看长度的向量
                img_emb = F.normalize(img_emb, dim=1)
                txt_emb = F.normalize(txt_emb, dim=1)

                # 2. 计算点积相似度矩阵 (并除以 0.07 这个 AI 界的黄金 Temperature 参数)
                sim_matrix = torch.matmul(img_emb, txt_emb.T) / 0.07

                # 3. 对角线上的元素互为正样本，其余全为负样本 (强迫同一患者的图文拉近，不同患者的推远)
                batch_curr_size = img_emb.size(0)
                contrastive_labels = torch.arange(batch_curr_size).to(device)
                
                loss_i2t = F.cross_entropy(sim_matrix, contrastive_labels) # 以图找文
                loss_t2i = F.cross_entropy(sim_matrix.T, contrastive_labels) # 以文找图
                loss_contrastive = (loss_i2t + loss_t2i) / 2.0

                # ==================================================
                # 【双擎融合】：最终的总 Loss
                # ==================================================
                total_train_loss = maxsup_loss + 0.5 * loss_contrastive

            # 反向传播与梯度缩放
            scaler.scale(total_train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 统计战况
            total_loss += total_train_loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            if batch_idx % 50 == 0:
                current_acc = 100 * correct_preds / total_samples
                print(f"⏳ Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] "
                      f"| 综合Loss: {total_train_loss.item():.4f} "
                      f"| (分类: {maxsup_loss.item():.3f}, 对比: {loss_contrastive.item():.3f}) "
                      f"| 批次准确率: {current_acc:.1f}%")

        epoch_time = time.time() - start_time
        epoch_acc = 100 * correct_preds / total_samples
        print("-" * 70)
        print(f"✅ Epoch {epoch+1} 结束! 耗时: {epoch_time:.1f}s | 平均综合 Loss: {total_loss/len(train_loader):.4f} | 真实影像总准确率: {epoch_acc:.2f}%")
        print("-" * 70)

    # 3. 固化战斗成果
    save_path = "./checkpoints/real_medical_model_sota.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n🏆 炼丹完美收官！ResNet 与 BERT 已在潜空间完成史诗级会师！")
    print(f"💾 实战权重已极其安全地保存至: {save_path}")

if __name__ == "__main__":
    train_on_real_data(epochs=2, batch_size=8)
