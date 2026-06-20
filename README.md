# Low-Resource-Medical-LLM
本仓库记录了我最终目标进入大厂医疗AI军团短期目标冲刺上海中医药大学中医人工智能研究生的进阶之路。通过在RTX 4050上复现BioMedLM的微调过程，深度拆解神经网络线性层Affine变换原理，探索y=xW 
T+b算子在‘证候特征对齐’中的数学本质”
“项目不仅包含完整的QLoRA实验记录，更致力于将中医《伤寒论》等经典文献知识图谱化，通过检索增强（RAG）技术抑制模型幻觉。核心攻关点在于：在资源受限环境下，实现中西医多模态数据（电子病历、影像、中医望诊信号）的特征空间映射与对齐。
项目正在迭代中首阶段侧重于医学数据集的清洗与QLoRA环境的搭建。
核心技术壁垒 (The Moat)
本项目的核心目标并非盲目追求参数规模，而是在资源受限环境下，通过极其精密的算法“微操”，强行纠正黑盒 AI 的视觉幻觉与数据偏见。
1. 极端算力下的算法微操 (Micro-surgery for RTX 4050)
在仅有 6GB 显存的设备上，彻底摒弃了暴力的全参数微调。通过 requires_grad=False 冻结预训练视觉底座（ResNet-50）80% 参数。重写 img_proj 投影层，将 2048 维特征极致降维至 256 维。结合 AMP (自动混合精度)，成功将包含百万级参数的双模态重型机甲，单 Epoch 训练耗时压榨至 100 秒以内。引入 InfoNCE 潜空间对齐与 Cross-Attention 交叉注意力，实现了医学影像与临床主诉（中/西医文本）在 256 维潜空间的完美握手。
2. 双轨制仲裁引擎 (Dual-Track Arbitration Engine)
打破了单凭视觉网络“盲猜”的困局，引入了 中医 GraphRAG 循证链条。直觉脑 (Connectionism)：基于 ResNet 提取底层影像特征，但受限于数据集极易产生“过度自信 (Overconfidence)”。理智脑 (Symbolic AI)：基于 NetworkX 构建的中西医知识图谱，能精准通过“盗汗、潮热”等文本主诉锁定中医证型（如：肺阴亏耗证），并映射至西医疾病。Logit 铁腕干预：当视觉黑盒与图谱逻辑发生冲突时，通过 Softmax Temperature Scaling (T=5.0) 打碎神经网络的盲目傲慢，并在 Logit 层注入动态 Boost 惩罚权重，实现教科书级别的“AI 幻觉逆转”。
3. 可解释性 AI 视觉追踪 (XAI Grad-CAM)
深度学习在医疗领域最致命的缺陷是“捷径学习 (Shortcut Learning)”。系统在 ResNet 最后一层注册钩子（Hook）提取反向传播梯度。通过提取梯度绝对值修复了网络深层的梯度抵消问题，结合动态 Alpha 遮罩，实时渲染 Grad-CAM 视觉注意力热力图。实战意义：成功捕获了底层模型“偷看 X 光片边缘字母 R 盲猜肺炎”的作弊行为，将黑盒决策逻辑完全暴露，为 GraphRAG 的后续介入提供了法理依据。
## 技术关键词 Tags
#Medical-AI #QLoRA #RTX4050-Optimization #Responsible-AI #BioBERT #ResNet #NLP<img width="2549" height="1397" alt="ScreenShot_2026-06-20_153354_633" src="https://github.com/user-attachments/assets/550da823-833c-4f4c-b7c0-3ad6fcb680f1" />

