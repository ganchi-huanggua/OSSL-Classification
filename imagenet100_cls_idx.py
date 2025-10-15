import json

def get_imagenet100_class_ids(imagenet100_folder_names, json_path):
    """
    根据 ImageNet-100 文件夹名列表，从 ImageNet_class_index.json 中获取对应的类号
    :param imagenet100_folder_names: list，ImageNet-100 的文件夹名列表（如 ["n01669191", "n01871265"]）
    :param json_path: str，ImageNet_class_index.json 文件的路径（如 "./ImageNet_class_index.json"）
    :return: dict，key 为文件夹名，value 为对应的类号（未匹配到的 value 为 None）
    """
    # 1. 加载 ImageNet_class_index.json 并建立“文件夹名→类号”的映射
    folder_to_class_id = {}
    with open(json_path, 'r', encoding='utf-8') as f:
        class_index = json.load(f)  # class_index 格式：{"类号": ["文件夹名", "类名"]}
    
    # 遍历 json 数据，反转键值对（原：类号→[文件夹名, 类名]；目标：文件夹名→类号）
    for class_id, (folder_name, class_name) in class_index.items():
        folder_to_class_id[folder_name] = class_id  # 文件夹名作为 key，类号作为 value

    # 2. 遍历 ImageNet-100 文件夹名列表，匹配对应的类号
    result = {}
    unmatched = []  # 记录未匹配到的文件夹名（用于后续检查）
    for folder_name in imagenet100_folder_names:
        if folder_name in folder_to_class_id:
            result[folder_name] = folder_to_class_id[folder_name]
        else:
            result[folder_name] = None
            unmatched.append(folder_name)
    
    # 3. 打印未匹配的文件夹名（提示用户检查）
    if unmatched:
        print(f"注意：以下 {len(unmatched)} 个文件夹名在 {json_path} 中未找到对应类号：")
        for fn in unmatched:
            print(f"  - {fn}")
    else:
        print(f"成功：所有 {len(imagenet100_folder_names)} 个文件夹名都匹配到了对应的类号！")
    
    return result


# ------------------- 示例使用 -------------------
if __name__ == "__main__":
    # 1. 替换为你的 ImageNet-100 文件夹名列表（示例：包含之前提到的 box_turtle、tusker 等）
    your_imagenet100_folders = [
 "n01558993", "n01770393", "n02087046", "n02097130", "n02106030", "n02120505", "n02364673", "n02906734", "n03255030", "n03498962", "n03733281", "n03854065", "n04004767", "n04311004", "n04443257", "n04554684", "n07693725",
    "n01601694", "n01855672", "n02088632", "n02097298", "n02106166", "n02125311", "n02484975", "n02909870", "n03272010", "n03530642", "n03759954", "n03929855", "n04026417", "n04325704", "n04458633", "n04591157", "n07711569",
    "n01669191", "n01871265", "n02093256", "n02099267", "n02107142", "n02128385", "n02489166", "n03085013", "n03291819", "n03623198", "n03775071", "n03930313", "n04065272", "n04336792", "n04483307", "n04592741", "n07753592",
    "n01751748", "n02018207", "n02093754", "n02100877", "n02110341", "n02133161", "n02708093", "n03124170", "n03337140", "n03649909", "n03814639", "n03954731", "n04200800", "n04346328", "n04509417", "n04606251", "n11879895",
    "n01755581", "n02037110", "n02094114", "n02104365", "n02114855", "n02277742", "n02747177", "n03127747", "n03450230", "n03710721", "n03837869", "n03956157", "n04209239", "n04380533", "n04515003", "n07583066",
    "n01756291", "n02058221", "n02096177", "n02105855", "n02120079", "n02325366", "n02835271", "n03160309", "n03483316", "n03717622", "n03838899", "n03983396", "n04235860", "n04428191", "n04525305", "n07613480"
    ]
    
    # 2. 替换为你的 ImageNet_class_index.json 文件路径
    your_json_path = "/home/lhz/data/imagenet100/ImageNet_class_index.json"  # 例如：放在代码同目录下
    
    # 3. 调用函数获取结果
    class_id_result = get_imagenet100_class_ids(your_imagenet100_folders, your_json_path)
    
    cids = []
    
    # 4. 打印结果（按“文件夹名→类号”格式输出，方便查看）
    print("\nImageNet-100 文件夹名对应类号：")
    for folder, cid in class_id_result.items():
        print(f"文件夹名：{folder} → 类号：{cid}")
        cids.append(int(cid))
        
    print(cids)