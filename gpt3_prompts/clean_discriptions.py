import json
import re
import os
# 1. 读取原始JSON文件（请确保文件路径正确，若文件在同级目录可直接用文件名）
with open('./gpt3_prompts/CuPL_prompts_flowers102.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 定义清洗规则所需的辅助内容（可根据实际需求修改）
# 要过滤的敏感词/无意义词列表
stop_words = {'无效', '无意义', '测试', '占位', 'TODO'}
# 允许保留的标点符号（其他标点会被自动移除）
allowed_punctuation = {'.', ',', ';', ':', '\\', '(', ')'}
# 用于判断“符号开头”的标点集合（避免描述以符号起始）
punctuation_set = set('.,;!:?()[]{}""''`~@#$%^&*_-+=|\\<>/')

# 3. 遍历每个类，对其对应的描述列表逐个清洗
# 先统计原始总描述数（避免后续遍历丢失原始数据）
original_total = 0
for class_name in data:
    original_total += len(data[class_name])

for class_name in data:
    current_descriptions = data[class_name]
    cleaned_descriptions = []  # 存储当前类清洗后的描述
    
    for desc in current_descriptions:
        # 跳过空值或完全由空格组成的描述
        if not desc or desc.strip() == '':
            continue
        
        # 3.1 空格处理：只删除开头空格，保留句子中间和末尾空格
        # 用lstrip()仅去除左侧空格，右侧和中间空格不变
        # 3.2 过滤“符号开头”的描述（处理完开头空格后判断）
        if desc[0] in punctuation_set or desc[0] == ' ':
            continue  # 若开头是符号，直接排除该描述
        # 3.1 空格处理：只删除开头空格，保留句子中间和末尾空格
        # 用lstrip()仅去除左侧空格，右侧和中间空格不变
        
        cleaned = desc.strip()
        # 3.3 格式规范化：统一首字母大写（仅针对非空描述）
        if len(cleaned) > 1:
            # 首字母转大写，后面字符保持原格式（避免全小写转大写破坏语义）
            cleaned = cleaned[0].upper() + cleaned[1:]
        elif len(cleaned) == 1:
            cleaned = cleaned.upper()  # 单个字符时直接转大写
        
        # 3.4 内容过滤：移除不允许的标点符号
        filtered_chars = []
        for char in cleaned:
            # 保留字母、数字、空格、允许的标点
            if char.isalnum() or char.isspace() or char in allowed_punctuation:
                filtered_chars.append(char)
        cleaned = ''.join(filtered_chars)
        
        # 3.5 过滤包含敏感词的描述
        has_stop_word = False
        for word in stop_words:
            if word in cleaned:
                has_stop_word = True
                break
        if has_stop_word:
            continue
        
        # 3.6 过滤过短无意义描述（长度小于5字符的排除，可调整阈值）
        if len(cleaned) < 50:
            continue
        
        # 3.7 将符合要求的描述加入列表（后续去重）
        cleaned_descriptions.append(cleaned)
    
    # 3.8 去重：用set去重后转回列表，保持类名对应结构不变
    data[class_name] = list(set(cleaned_descriptions))

# 4. 保存清洗后的数据到新JSON文件（避免覆盖原始文件）
with open('./gpt3_prompts/cleaned_CuPL_prompts_flowers102.json', 'w', encoding='utf-8') as f:
    # ensure_ascii=False：保留中文等非ASCII字符；indent=2：格式化显示，便于查看
    json.dump(data, f, ensure_ascii=False, indent=2)

# 5. 统计并输出清洗结果
cleaned_total = 0
for class_name in data:
    cleaned_total += len(data[class_name])

print("=" * 50)
print("清洗完成！")
print(f"原始描述总数：{original_total}")
print(f"清洗后描述总数：{cleaned_total}")
print(f"过滤掉的描述数：{original_total - cleaned_total}")
print(f"清洗后文件路径：./cleaned_class_descriptions.json")
print("=" * 50)