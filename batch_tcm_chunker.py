import os
import re
import json
from pathlib import Path

def build_sliding_window_chunks(input_dir, output_path, chunk_size=300, overlap=50):
    print(f"🚀 [V3引擎 - 智能编码识别] 正在启动工业级滑动窗口切片: {input_dir}")
    all_chunks = []
    file_count = 0
    
    # 递归遍历目录
    for filepath in Path(input_dir).rglob('*.txt'):
        book_name = filepath.stem
        
        # 【核心修复：双重编码智能识别】
        raw_text = ""
        try:
            # 优先尝试现代标准的 UTF-8 编码读取
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except UnicodeDecodeError:
            try:
                # 如果报错，说明是老系统的中文文件，无缝切换到 GB18030 国标编码读取
                with open(filepath, 'r', encoding='gb18030') as f:
                    raw_text = f.read()
            except Exception:
                continue # 如果还是失败，说明文件损坏，直接跳过
        except Exception:
            continue
            
        if not raw_text:
            continue
            
        file_count += 1
        # 清洗：抹平杂乱排版
        text_cleaned = re.sub(r'\s+', '', raw_text)
        
        # 滑动窗口切片
        start = 0
        text_len = len(text_cleaned)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text_cleaned[start:end]
            
            if len(chunk_text) >= 50: 
                all_chunks.append({
                    "book": book_name,
                    "text": chunk_text,
                    "length": len(chunk_text)
                })
            start += (chunk_size - overlap)
            
    print(f"\n📚 扫描完毕！共成功处理 {file_count} 本古籍。")
    print(f"⚙️ V3 引擎发力，共计生成了 {len(all_chunks)} 个纯净中文知识块...")
    
    # 统一转换并保存为 UTF-8 格式的 JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
    print(f"✅ V3 切片完成！已保存至 {output_path}")

if __name__ == "__main__":
    input_directory = "data/TCM-Ancient-Books" 
    output_file = "data/kg_data/tcm_massive_chunks.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    build_sliding_window_chunks(input_directory, output_file)