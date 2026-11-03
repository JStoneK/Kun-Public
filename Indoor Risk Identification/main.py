"""
室内风险识别 - 基于 Qwen2.5-VL 本地推理
依赖安装：pip install transformers torch torchvision qwen-vl-utils accelerate
"""

import os
import re
import json
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# ==================== 模块 0：场景 Prompt 映射 ====================

SCENE_PROMPTS = {
    "bathroom": """你是专业浴室安全巡检员，对输入的浴室图片做安全隐患检测，严格遵守以下规则，只输出JSON，禁止额外解释文字。
1. 需要重点观测区域：洗手台台面、地面瓷砖、洗手台边缘、马桶周边、过道通道、绿植盆栽、各类洗护瓶罐
2. 浴室安全判定规则：
规则1：地面散落洗护瓶、绿植、杂物 → 滑倒绊倒风险（等级：中）
规则2：地面存在积水、水渍 → 滑倒摔伤风险（等级：高）
规则3：洗护瓶、香薰等物品摆放超出洗手台边缘 → 物品坠落砸脚、堵塞下水风险（等级：低）
规则4：地面大面积堆放物品、堵塞过道 → 通行受阻绊倒风险（等级：高）
规则5：花盆绿植倾斜、倾倒 → 泥土积水打滑风险（等级：中）
规则6：台面洗护容器倾倒、液体泄漏 → 地面湿滑摔倒风险（等级：高）
规则7：画面无以上任意情况 → 无安全风险
3. 固定输出JSON字段，字段内容严格贴合图片实际：
{
"has_risk": true/false,
"risk_type": "滑倒绊倒/物品坠落/通行受阻/无风险",
"risk_objects": ["隐患涉及的物品名称"],
"location": "隐患所在位置（地面/洗手台边缘/过道）",
"risk_level": "低/中/高",
"judge_reason": "对应哪一条判定规则",
"suggestion": "简短整改建议"
}
""",

    "bedroom": """你是专业卧室安全巡检员，对输入的卧室图片开展安全隐患检测，严格遵循下方全部判定规则，仅输出标准JSON文本，禁止输出任何多余解释、说明文字。
1. 重点观测区域：卧室木地板地面、床边地毯、过道通行区域、床头柜、床沿、衣柜隔断周边、门窗走道
2. 卧室安全隐患判定规则：
规则1：地面存在破碎玻璃、尖锐碎片 → 划伤扎脚风险（风险等级：高）
规则2：地面、地毯散落书本、摆件、杂物 → 绊倒、磕碰风险（风险等级：中）
规则3：床头柜水杯、重物贴近边缘，极易掉落 → 砸伤脚部、物品碎裂风险（风险等级：低）
规则4：过道大面积堆放杂物、碎片堵塞通行路线 → 绊倒、紧急疏散受阻风险（风险等级：高）
规则5：床上、床沿织物大面积垂落地面、遮挡过道 → 绊倒打滑风险（风险等级：低）
规则6：画面无上述任意隐患情况 → 无安全风险
3. 强制输出固定JSON字段，内容严格匹配图片实景：
{
"has_risk": true/false,
"risk_type": "划伤扎脚/绊倒磕碰/物品坠落/疏散受阻/无风险",
"risk_objects": ["隐患对应的物品名称"],
"location": "隐患所在位置（地面过道/床边地毯/床头柜/床沿）",
"risk_level": "低/中/高",
"judge_reason": "对应本条判定规则编号",
"suggestion": "简短可落地的整改建议"
}
""",

    "corridor": """你是专业室内消防通道巡检员，对走廊图片开展安全隐患检测，严格遵守所有判定规则，仅输出纯JSON文本，禁止添加任何多余解释、说明文字。
1. 需要重点观测区域：消防出口（FIRE EXIT）红色大门、走廊主通行过道、靠墙桌面、地面电线插排、天花板悬挂绿植、过道置物架
2. 走廊安全隐患判定规则：
规则1：桌子、大水桶、置物架等大件物品占用过道、遮挡消防逃生门 → 疏散通道堵塞，火灾逃生受阻（风险等级：高）
规则2：地面裸露插线板、外露电线线缆 → 绊倒、漏电触电风险（风险等级：中）
规则3：天花板悬挂盆栽无防护固定 → 高空坠物砸伤行人（风险等级：低）
规则4：过道大面积堆放杂物，完全阻断通往消防门的路线 → 紧急疏散失效（风险等级：高）
规则5：画面无以上任意隐患，过道通畅、消防门无遮挡 → 无安全风险
3. 强制输出固定JSON字段，内容严格匹配图片实景：
{
"has_risk": true/false,
"risk_type": "通道堵塞/触电绊倒/高空坠物/无风险",
"risk_objects": ["隐患涉及物品名称"],
"location": "隐患所在位置（走廊过道/消防门前/地面/天花板）",
"risk_level": "低/中/高",
"judge_reason": "对应本条判定规则编号",
"suggestion": "简短可落地的整改建议"
}
""",

    "kitchen": """你是专业厨房安全巡检员，对输入的厨房图片开展安全隐患检测，严格遵守全部判定规则，仅输出纯JSON文本，禁止输出任何多余解释、说明文字。
1. 重点观测区域：燃气灶灶台、中岛操作台、厨房地面过道、橱柜周边、电器柜体、灶台旁储物容器
2. 厨房安全隐患判定规则：
规则1：燃气灶区域出现明火、火焰燃烧 → 厨房火灾爆炸风险（风险等级：高）
规则2：塑料瓶、纸质调料罐、木质容器紧贴灶台明火区域 → 易燃物引燃扩大火情（风险等级：高）
规则3：厨房地面散落瓶罐、杂物、厨具 → 滑倒、绊倒磕碰风险（风险等级：中）
规则4：大件家具、物品遮挡厨房通行过道，阻断逃生路线 → 紧急疏散受阻（风险等级：中）
规则5：画面无以上任意隐患，灶台无明火、地面整洁通畅 → 无安全风险
3. 强制输出固定JSON字段，内容严格匹配图片实景：
{
"has_risk": true/false,
"risk_type": "火灾隐患/滑倒绊倒/疏散受阻/无风险",
"risk_objects": ["隐患涉及物品名称"],
"location": "隐患所在位置（灶台区域/厨房地面/过道操作台）",
"risk_level": "低/中/高",
"judge_reason": "对应本条判定规则编号",
"suggestion": "简短可落地的整改建议"
}
"""
}

# ==================== 模块 1：本地 VLM 加载 & 单图推理 ====================

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"  # 本地缓存路径或 HuggingFace 模型名

def load_model(model_path=MODEL_PATH):
    """加载本地 Qwen2.5-VL 模型，4bit 量化适配 6GB 显存（RTX 4050）"""
    from transformers import BitsAndBytesConfig
    print(f"[INFO] 正在加载模型: {model_path}（4bit 量化）")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("[INFO] 模型加载完成（4bit 量化，显存占用约 3.5GB）")
    return model, processor


def infer_image(model, processor, image_path, prompt):
    """单图推理接口，返回模型原始文本输出"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


# ==================== 模块 2：数据集遍历逻辑 ====================

def get_scene_images(dataset_dir):
    """遍历 datasets 下 4 个场景子文件夹，返回 {场景名: [图片路径列表]}"""
    scene_images = {}
    # 实际文件夹名到 prompt 键的映射
    folder_to_scene = {
        "Corridor": "corridor",
        "bathroom": "bathroom",
        "bedroom": "bedroom",
        "kitchen": "kitchen",
    }
    for folder_name, scene_name in folder_to_scene.items():
        scene_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(scene_path):
            print(f"[WARN] 场景文件夹不存在: {scene_path}")
            continue
        images = []
        for fname in sorted(os.listdir(scene_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(scene_path, fname))
        scene_images[scene_name] = images
    return scene_images


# ==================== 模块 3：结果清洗 + JSON 保存 ====================

def extract_json(raw_text):
    """从模型返回文本中提取纯 JSON 字符串"""
    # 尝试匹配第一个 { 到最后一个 } 之间的内容
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if match:
        json_str = match.group(0)
        return json.loads(json_str)
    return None


def save_report(report_data, report_dir, image_path):
    """保存 JSON 报告到 report/对应场景/ 文件夹"""
    os.makedirs(report_dir, exist_ok=True)
    image_name = Path(image_path).stem
    report_path = os.path.join(report_dir, f"{image_name}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    return report_path


# ==================== 主流程 ====================

def main():
    # 路径配置
    project_root = Path(__file__).parent
    dataset_dir = os.path.join(project_root, "datasets")
    report_dir = os.path.join(project_root, "report")

    # 1. 加载模型
    model, processor = load_model()

    # 2. 获取所有场景图片
    scene_images = get_scene_images(dataset_dir)

    # 3. 遍历推理
    total = sum(len(imgs) for imgs in scene_images.values())
    processed = 0

    for scene_name, images in scene_images.items():
        if not images:
            print(f"[INFO] 场景 {scene_name} 无图片，跳过")
            continue

        prompt = SCENE_PROMPTS[scene_name]
        scene_report_dir = os.path.join(report_dir, scene_name)

        print(f"\n{'='*50}")
        print(f"[INFO] 开始处理场景: {scene_name} ({len(images)} 张图片)")
        print(f"{'='*50}")

        for img_path in images:
            processed += 1
            print(f"\n[{processed}/{total}] 推理: {img_path}")

            try:
                # 调用模型推理
                raw_output = infer_image(model, processor, img_path, prompt)
                print(f"[RAW] {raw_output[:100]}...")

                # 提取 JSON
                report_data = extract_json(raw_output)
                if report_data is None:
                    print(f"[ERROR] JSON 提取失败，跳过: {img_path}")
                    continue

                # 保存报告
                report_path = save_report(report_data, scene_report_dir, img_path)
                print(f"[OK] 报告已保存: {report_path}")

            except Exception as e:
                print(f"[ERROR] 处理失败 {img_path}: {e}")
                continue

    print(f"\n{'='*50}")
    print(f"[DONE] 全部处理完成，共处理 {processed}/{total} 张图片")
    print(f"[DONE] 报告输出目录: {report_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()