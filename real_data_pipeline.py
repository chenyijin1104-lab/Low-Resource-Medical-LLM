import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer
import matplotlib.pyplot as plt

# 禁用 tokenizer 并行警告，保持终端清爽
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. 真实的医学多模态 Dataset (后勤弹药装填机)
# ==========================================
class KaggleXRayMultimodalDataset(Dataset):
    def __init__(self, data_dir="./chest_xray/train", tokenizer_name='hfl/chinese-macbert-base', max_len=128):
        """
        data_dir: 指向 kaggle 解压后的 train 目录
        """
        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.samples = []
        
        # 工业级图像预处理流水线（专为 4050 显存优化）
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),               # 强制统一缩放，防止显存爆炸
            transforms.Grayscale(num_output_channels=1), # 强制转为单通道灰度图
            transforms.ToTensor(),                       # 转为张量，数值缩放至 [0, 1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1, 1]，加速神经网络收敛
        ])

        # 真实的临床体征知识库（动态匹配文本）
        self.clinical_texts = {
            0: [ # 0 代表 NORMAL (正常)
                "常规入职体检，患者无发热、无咳嗽咳痰，自诉无胸闷气急。听诊双肺呼吸音清，未闻及干湿性啰音。",
                "患者体温正常，无呼吸道感染症状。查体胸廓对称，气管居中，心音有力，心律齐。",
                "门诊复查，目前已无不适主诉。肺部叩诊呈清音，听诊无异常发现。"
            ],
            1: [ # 1 代表 PNEUMONIA (肺炎)
                "患者突发高热39度，伴寒战，剧烈咳嗽，咳黄绿色脓痰。右下肺听诊可闻及明显细湿啰音。",
                "主诉胸痛伴呼吸困难2天，深呼吸时加重。白细胞计数显著升高，叩诊右侧下肺呈浊音。",
                "患儿持续低热，精神萎靡，伴阵发性痉挛性咳嗽。听诊双肺呼吸音粗糙，可闻及管状呼吸音。"
            ]
        }

        # 遍历硬盘，收集所有图片的路径
        print(f"🔍 正在扫描真实医疗数据目录: {data_dir} ...")
        for label, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.samples.append({
                        'img_path': os.path.join(class_dir, img_name),
                        'label': label
                    })
        
        if len(self.samples) == 0:
            raise FileNotFoundError(f"❌ 严重错误：在 {data_dir} 下没有找到任何图片！请确认你把解压后的 chest_xray 文件夹放对了位置。")
        else:
            print(f"✅ 扫描完毕！共找到 {len(self.samples)} 张真实胸透 X 光片。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 处理图像 (通过 torchvision 转化为张量)
        # 强制先转RGB再灰度，这是计算机视觉里防报错的黄金法则，防止某些奇奇怪怪的图片格式
        img = Image.open(sample['img_path']).convert('RGB') 
        img_tensor = self.transform(img)

        # 2. 处理文本 (随机抽一条符合该疾病特征的病历进行 Tokenize)
        text = random.choice(self.clinical_texts[sample['label']])
        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'images': img_tensor,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sample['label'], dtype=torch.long),
            'raw_text': text # 保留纯文本，仅用于展示
        }

# ==========================================
# 2. 阅兵大典：验证数据流水线是否通畅
# ==========================================
def test_data_pipeline():
    # 这里是你的 Kaggle 文件夹路径
    train_dir = "./chest_xray/train"
    
    if not os.path.exists(train_dir):
        print(f"❌ 路径错误：找不到 {train_dir}")
        print("请确保你解压出来的文件夹名字叫 chest_xray，并且和当前这个 python 文件在同一个目录下！")
        return

    # 实例化我们的多模态数据集
    dataset = KaggleXRayMultimodalDataset(data_dir=train_dir)
    
    # 启动工业级 DataLoader (Batch Size 设为 4，保证显存绝对安全)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 抓取一个 Batch 的数据进行质检
    print("\n📦 正在抽取第一个 Batch 的真实数据进行预览...")
    batch = next(iter(dataloader))
    
    images = batch['images']
    labels = batch['labels']
    texts = batch['raw_text']
    
    # 打印给算法工程师看的维度信息
    print("-" * 50)
    print(f"📊 图像张量形状 (Batch, Channel, Height, Width): {images.shape}")
    print(f"📊 标签张量形状: {labels.shape}")
    print("-" * 50)
    
    # 准备弹出画板
    # 强制设置 Windows 系统的中文字体，防止汉字变成方块
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle('真实医疗影像多模态流水线预览 (Kaggle Dataset)', fontsize=16)
    
    for i in range(4):
        # 把张量变回 Numpy 矩阵给 matplotlib 画图
        img_np = images[i].numpy().squeeze()
        
        # 因为在 DataLoader 里做了 [-1, 1] 的归一化，现在要反向操作变回 [0, 1]，否则图片会发黑或失真！
        img_np = img_np * 0.5 + 0.5 
        
        label_str = "肺炎 (PNEUMONIA)" if labels[i] == 1 else "正常 (NORMAL)"
        color = 'red' if labels[i] == 1 else 'green'
        
        axes[i].imshow(img_np, cmap='gray')
        axes[i].set_title(label_str, color=color, pad=10)
        axes[i].axis('off')
        
        # 截断显示长文本，防止挡住图片
        display_text = texts[i][:25] + "..." if len(texts[i]) > 25 else texts[i]
        axes[i].text(0, 275, display_text, fontsize=10, wrap=True, color='blue')
        
    plt.tight_layout()
    plt.show()
    print("🎉 弹窗已生成！如果看到了真实的肺部 X 光片，说明后勤补给线全线贯通！")

if __name__ == "__main__":
    test_data_pipeline()
