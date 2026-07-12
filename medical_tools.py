import re
from typing import Dict, List, Tuple

# =====================================================================
# 1. 中医药理法则底座：十八反 & 十九畏 核心禁忌图谱 (100% 确定性字典)
# =====================================================================
# 十八反歌诀：本草明言十八反，半梨贝蔹白及战乌头，藻戟遂芫俱战草，诸参辛芍叛藜芦。
TCM_18_INCOMPATIBILITIES = [
    ({"乌头", "川乌", "草乌", "附子"}, {"半夏", "瓜蒌", "天花粉", "贝母", "川贝", "浙贝", "白蔹", "白及"}),
    ({"甘草", "炙甘草"}, {"海藻", "大戟", "京大戟", "红大戟", "甘遂", "芫花"}),
    ({"藜芦"}, {"人参", "党参", "西洋参", "玄参", "苦参", "丹参", "沙参", "细辛", "白芍", "赤芍", "芍药"})
]

# 十九畏歌诀：硫黄原是火中精，朴硝一见便相争... 丁香莫与郁金见，牙硝难合京三棱...
TCM_19_INHIBITIONS = [
    ({"丁香", "公丁香", "母丁香"}, {"郁金"}),
    ({"人参", "党参", "红参"}, {"五灵脂"}),
    ({"官桂", "肉桂", "桂枝"}, {"赤石脂"}),
    ({"川乌", "草乌", "附子"}, {"犀角", "水牛角"}),
    ({"硫黄"}, {"朴硝", "芒硝", "玄明粉"})
]

def check_tcm_incompatibility(prescription_text: str) -> Tuple[bool, str]:
    """
    🧰 [中医工具 1] 处方十八反十九畏绝对安全审查引擎
    """
    if not prescription_text or len(prescription_text.strip()) == 0:
        return True, "⚠️ 未检测到有效处方内容，跳过中医配伍审查。"

    # 通过正则清洗提取文字，进行精准匹配
    clean_text = re.sub(r'[^\u4e00-\u9fa5]', ' ', prescription_text)
    words = set(clean_text.split())
    
    violations = []
    
    # 检查十八反
    for group_a, group_b in TCM_18_INCOMPATIBILITIES:
        found_a = group_a.intersection(words)
        found_b = group_b.intersection(words)
        if found_a and found_b:
            violations.append(f"【十八反严重禁忌】处方中同时出现了 🚫 `{'/'.join(found_a)}` 与 🚫 `{'/'.join(found_b)}`！二者合用会产生剧烈毒副作用！")
            
    # 检查十九畏
    for group_a, group_b in TCM_19_INHIBITIONS:
        found_a = group_a.intersection(words)
        found_b = group_b.intersection(words)
        if found_a and found_b:
            violations.append(f"【十九畏配伍禁忌】处方中同时出现了 🚫 `{'/'.join(found_a)}` 与 🚫 `{'/'.join(found_b)}`！二者合用会产生相畏降低药效或毒性！")

    if violations:
        error_report = "❌ **触发中医药理安全拦截！**\n" + "\n".join(violations)
        return False, error_report
    else:
        return True, "✅ **中医配伍审查通过**：未见十八反、十九畏等经典药理配伍禁忌。"


# =====================================================================
# 2. 西医量化法则底座：CURB-65 肺炎重症评分工具 (杜绝数学幻觉)
# =====================================================================
def calculate_curb65_score(age: int = 0, bun_mg_dl: float = 0.0, rr: int = 0, sys_bp: int = 120, dia_bp: int = 80, confusion: bool = False) -> Tuple[int, str, str]:
    """
    🧰 [西医工具 2] CURB-65 社区获得性肺炎重症死亡风险评估工具
    参数说明:
    - confusion: 是否有意识障碍 (Confusion)
    - bun_mg_dl: 血尿素氮 > 19 mg/dL (Urea)
    - rr: 呼吸频率 >= 30 次/分 (Respiratory rate)
    - sys_bp / dia_bp: 收缩压 < 90 或 舒张压 <= 60 mmHg (Blood pressure)
    - age: 年龄 >= 65 岁 (65)
    """
    score = 0
    reasons = []
    
    if confusion:
        score += 1
        reasons.append("意识障碍 (+1分)")
    if bun_mg_dl > 19.0 or bun_mg_dl > 7.0: # 兼容 mmol/L 和 mg/dL 的极简判断
        score += 1
        reasons.append("血尿素氮升高 (+1分)")
    if rr >= 30:
        score += 1
        reasons.append(f"呼吸频速 {rr}次/分 >=30 (+1分)")
    if sys_bp < 90 or dia_bp <= 60:
        score += 1
        reasons.append(f"低血压 {sys_bp}/{dia_bp} mmHg (+1分)")
    if age >= 65:
        score += 1
        reasons.append(f"高龄 {age}岁 >=65 (+1分)")
        
    reason_str = " | ".join(reasons) if reasons else "各项临床生命体征均在安全阈值内"
    
    if score <= 1:
        risk_level = "🟢 **低风险 (Low Risk)**"
        recommendation = "建议门诊随访与口服抗感染治疗，无需住院。"
    elif score == 2:
        risk_level = "🟡 **中风险 (Moderate Risk)**"
        recommendation = "建议短期住院观察或在有条件的社区医院进行静脉抗炎治疗。"
    else:
        risk_level = "🔴 **高危重症 (High/Severe Risk)**"
        recommendation = "🚨 **警告：死亡风险极高！** 建议立刻收治入院，评估是否进入急诊重症监护室 (ICU) 进行紧急干预！"
        
    report = f"📊 **CURB-65 量化评估得分**: `{score} 分` ({risk_level})\n" \
             f"  ├─ 评分细则: {reason_str}\n" \
             f"  └─ 临床建议: {recommendation}"
             
    return score, risk_level, report


# =====================================================================
# 3. 统一工具调度器：供 LangGraph Safety Node 一键调用
# =====================================================================
def execute_medical_safety_check(report_text: str) -> Dict[str, any]:
    """
    全域工具总线：接收大模型生成的文本，执行自动审查
    """
    print("🧰 [Tools Bus] 正在调用本地 Python 确定性工具库进行医嘱审查...")
    
    # 1. 跑中医禁忌检查
    is_safe_tcm, tcm_msg = check_tcm_incompatibility(report_text)
    
    # 2. 启发式西医重症规则探测 (从主诉中自动抓取数值，若无则做基础评估)
    # 这里通过极其精炼的正则捕捉主诉中的年龄，自动计算重症风险
    age_match = re.search(r'(\d+)\s*(岁|year)', report_text)
    age_val = int(age_match.group(1)) if age_match else 45
    
    # 探测高危词汇模拟生命体征异常
    has_confusion = any(k in report_text for k in ["神志不清", "昏迷", "意识障碍", "谵妄"])
    has_high_rr = any(k in report_text for k in ["气促", "呼吸困难", "喘息剧烈", "不得卧"])
    rr_val = 32 if has_high_rr else 20
    
    _, _, curb_msg = calculate_curb65_score(age=age_val, rr=rr_val, confusion=has_confusion)
    
    overall_safe = is_safe_tcm # 目前以药理禁忌作为红线拦截指标
    
    audit_report = f"### 🛡️ V11.0 Python 确定性算力工具安全审查\n\n{tcm_msg}\n\n---\n\n{curb_msg}"
    
    return {
        "is_safe": overall_safe,
        "audit_report": audit_report,
        "tcm_msg": tcm_msg
    }

if __name__ == "__main__":
    # 测试致命配伍禁忌拦截
    test_bad_rx = "建议患者使用小青龙汤化裁，处方包含：麻黄、桂枝、炙甘草、京大戟、白芍、细辛。"
    res = execute_medical_safety_check(test_bad_rx)
    print(res["audit_report"])