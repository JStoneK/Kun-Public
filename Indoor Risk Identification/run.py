import os
import json
import re
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from collections import defaultdict

# ====================== 分场景专用巡检Prompt ======================
SCENE_PROMPTS = {
    "bathroom": """你是专业浴室安全巡检员，对输入的浴室图片开展安全隐患检测，可以参考以下规则，也可以自行判断，检查结果描述尽量详细，要全方面考虑。
1. 需要重点观测区域：洗手台台面、地面瓷砖、洗手台边缘、马桶周边、地面碎片
2. 浴室安全隐患判定：
    1：地面碎片→ 割伤风险（等级：高）
    1：地面散落洗护瓶、绿植、杂物 → 滑倒绊倒风险（等级：中）
    2：地面存在积水、水渍 → 滑倒摔伤风险（等级：高）
    3：洗护瓶、香薰等物品摆放超出洗手台边缘 → 物品坠落砸脚、堵塞下水风险（等级：低）
    4：地面大面积堆放物品、堵塞过道 → 通行受阻绊倒风险（等级：高）
    5：花盆绿植倾斜、倾倒 → 泥土积水打滑风险（等级：中）
    6：台面洗护容器倾倒、液体泄漏 → 地面湿滑摔倒风险（等级：高）
    7：组合风险，是否存在多个物品共同组成风险，例如：洗护瓶倾倒漏水+地面无防滑垫，叠加滑倒重伤风险（风险等级自行综合判断）
    8：应自己进行全方位考虑，发现图片中未列举的潜在浴室安全风险
    9：画面无以上任意隐患，地面干燥整洁、通道通畅、物品摆放规整 → 无安全风险
3. 重要输出要求：
- 若图片存在多处隐患，输出JSON数组包裹多条隐患对象；仅单条隐患直接输出单个对象；无风险输出单条对象
4. 强制输出固定JSON字段，描述尽量详细，内容严格匹配图片实景：
{
"has_risk": true/false,
"risk_type": "滑倒绊倒/物品坠落/通行受阻/无风险",
"risk_objects": ["隐患涉及的物品名称"],
"location": "隐患所在位置（地面/洗手台边缘/过道/马桶周边）",
"risk_level": "低/中/高",
"suggestion": "对于每个风险的可落地的整改建议"
}
""",
    "bedroom": """你是专业卧室安全巡检员，对输入的卧室图片开展安全隐患检测，可以参考以下规则，也可以自行判断，检查结果描述尽量详细，要全方面考虑。
1. 需要重点观测区域：卧室木地板地面、床边地毯、过道通行区域、床头柜、床沿、衣柜隔断周边、门窗走道
2. 卧室安全隐患判定：
    1：地面存在破碎玻璃、尖锐碎片 → 划伤扎脚风险（风险等级：高）
    2：地面、地毯散落书本、摆件、杂物 → 绊倒、磕碰风险（风险等级：中）
    3：床头柜水杯、重物贴近边缘，极易掉落 → 砸伤脚部、物品碎裂风险（风险等级：低）
    4：过道大面积堆放杂物、碎片堵塞通行路线 → 绊倒、紧急疏散受阻风险（风险等级：高）
    5：床上、床沿织物大面积垂落地面、遮挡过道 → 绊倒打滑风险（风险等级：低）
    6：组合风险，是否存在多个物品共同组成风险，例如：地面玻璃碎片+散落玩具，双重划伤绊倒隐患（风险等级自行综合判断）
    7：应自己进行全方位考虑，发现图片中未列举的潜在卧室安全风险
    8：画面无以上任意隐患，地面整洁、过道无遮挡、物品摆放安全 → 无安全风险
3. 重要输出要求：
- 若图片存在多处隐患，输出JSON数组包裹多条隐患对象；仅单条隐患直接输出单个对象；无风险输出单条对象
4. 强制输出固定JSON字段，描述尽量详细，内容严格匹配图片实景：
{
"has_risk": true/false,
"risk_type": "划伤扎脚/绊倒磕碰/物品坠落/疏散受阻/无风险",
"risk_objects": ["隐患对应的物品名称"],
"location": "隐患所在位置（地面过道/床边地毯/床头柜/床沿）",
"risk_level": "低/中/高",
"suggestion": "对于每个风险的可落地的整改建议"
}
""",
    "corridor": """你是专业室内消防通道巡检员，对走廊图片开展安全隐患检测，可以参考以下规则，也可以自行判断，检查结果描述尽量详细，要全方面考虑。
1. 需要重点观测区域：消防出口（FIRE EXIT）红色大门、走廊主通行过道、靠墙桌面、地面电线插排、天花板悬挂绿植、消防门前的矮置物架
2. 走廊安全隐患判定：
    1：桌子、大水桶、置物架等大件物品占用过道、遮挡消防逃生门 → 疏散通道堵塞，火灾逃生受阻（风险等级：高）
    2：过道大面积堆放杂物，完全阻断通往消防门的路线 → 紧急疏散失效（风险等级：高）
    3：地面裸露插线板、外露电线线缆 → 绊倒、漏电触电风险（风险等级：中）
    4：天花板悬挂盆栽无防护固定 → 高空坠物砸伤行人（风险等级：低）
    5：画面无以上任意隐患，过道通畅、消防门无遮挡、无触电风险 → 无安全风险
    6：注意消防门前的矮遮挡物（黑色置物品架子）
    7：组合风险，是否存在多个物品共同组成风险，例如：大水桶在桌面上，插线板在正下方，可能漏水触电（风险等级自己判断）
    8：应自己进行全方位考虑，发现潜在的风险
3. 重要输出要求：
- 若图片存在多处隐患，输出JSON数组包裹多条隐患对象；仅单条隐患直接输出单个对象；无风险输出单条对象
4. 强制输出固定JSON字段，描述尽量详细，内容严格匹配图片实景：
{
"has_risk": true/false,
"risk_type": "通道堵塞/触电绊倒/高空坠物/无风险",
"risk_objects": ["隐患涉及物品名称"],
"location": "走廊过道/消防门前/地面/天花板",
"risk_level": "低/中/高",
"suggestion": "对于每个风险的可落地的整改建议"
}
""",
    "kitchen": """你是专业厨房安全巡检员，对输入的厨房图片开展安全隐患检测，可以参考以下规则，也可以自行判断，检查结果描述尽量详细，要全方面考虑。
1. 重点观测区域：燃气灶灶台、中岛操作台、厨房地面过道、橱柜周边、电器柜体、灶台旁储物容器
2. 厨房安全隐患判定：
    1：燃气灶区域出现明火、火焰燃烧 → 厨房火灾爆炸风险（风险等级：高）
    2：塑料瓶、纸质调料罐、木质容器紧贴灶台明火区域 → 易燃物引燃扩大火情（风险等级：高）
    3：厨房地面散落瓶罐、杂物、厨具 → 滑倒、绊倒磕碰风险（风险等级：中）
    4：大件家具、物品遮挡厨房通行过道，阻断逃生路线 → 紧急疏散受阻（风险等级：中）
    5：组合风险，是否存在多个物品共同组成风险，例如：灶台明火旁堆放油桶+纸质包装袋，极易引发大火（风险等级自行综合判断）
    6：应自己进行全方位考虑，发现图片中未列举的潜在厨房安全风险
    7：画面无以上任意隐患，灶台无明火、地面整洁通畅、易燃物远离火源 → 无安全风险
3. 重要输出要求：
- 若图片存在多处隐患，输出JSON数组包裹多条隐患对象；仅单条隐患直接输出单个对象；无风险输出单条对象
4. 强制输出固定JSON字段，描述尽量详细，内容严格匹配图片实景：
{
"has_risk": true/false,
"risk_type": "火灾隐患/滑倒绊倒/疏散受阻/无风险",
"risk_objects": ["隐患涉及物品名称"],
"location": "灶台区域/厨房地面/过道操作台",
"risk_level": "低/中/高",
"suggestion": "对于每个风险的可落地的整改建议"
}
"""
}
# ====================== 路径&运行配置 ======================
MODEL_DIR = "./qwen2.5-vl-7b"
DATASET_ROOT = "./datasets"
REPORT_ROOT = "./report"

os.makedirs(REPORT_ROOT, exist_ok=True)

# ====================== 加载本地VLM模型 ======================
print("正在加载本地Qwen2.5-VL-7B离线模型...")
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()  # 推理模式，关闭dropout
print("模型加载完成，开始批量分场景巡检！\n")

# ====================== JSON清洗工具函数 ======================
def clean_json_text(raw_text: str) -> str:
    raw_text = re.sub(r"```(json)?", "", raw_text)
    raw_text = raw_text.strip()
    match = re.search(r"(\[.*\]|\{.*\})", raw_text, re.DOTALL)
    if match:
        return match.group(1)
    return raw_text

# 清洗模型生成的报告文本，去除markdown代码块、多余换行
def clean_report_text(raw_text: str) -> str:
    raw_text = re.sub(r"```.*?```", "", raw_text, flags=re.DOTALL)
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
    return raw_text.strip()

# ====================== 单图推理函数 ======================
def single_scene_inspect(img_path, scene_name):
    try:
        img = Image.open(img_path).convert("RGB")
        prompt = SCENE_PROMPTS[scene_name]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[img], return_tensors="pt")

        for k, v in inputs.items():
            if k == "pixel_values":
                inputs[k] = v.to("cuda", dtype=torch.float16)
            else:
                inputs[k] = v.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.1,
                top_p=0.3,
                repetition_penalty=1.05
            )
        raw_output = processor.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        json_str = clean_json_text(raw_output)

        raw_data = json.loads(json_str)
        if isinstance(raw_data, list):
            result_list = raw_data
        else:
            result_list = [raw_data]
        return result_list

    except json.JSONDecodeError as je:
        err_msg = f"JSON解析失败：{str(je)} 原始输出：{raw_output[:300]}"
        print(f"⚠️ {img_path} {err_msg}")
        return [{
            "has_risk": False,
            "risk_type": "输出JSON格式异常",
            "risk_objects": [],
            "location": "",
            "risk_level": "",
            "suggestion": "重新推理或更换图片"
        }]
    except Exception as e:
        err_msg = f"处理失败：{str(e)}"
        print(f"⚠️ {img_path} {err_msg}")
        return [{
            "has_risk": False,
            "risk_type": "处理异常",
            "risk_objects": [],
            "location": "",
            "risk_level": "",
            "suggestion": "检查图片文件格式与完整性"
        }]

# ====================== 批量遍历处理 ======================
total_img = 0
error_img = 0
risk_img = 0
all_scene_risk_data = []

for scene_dir_name in os.listdir(DATASET_ROOT):
    scene_path = os.path.join(DATASET_ROOT, scene_dir_name)
    if not os.path.isdir(scene_path) or scene_dir_name not in SCENE_PROMPTS:
        continue

    print(f"===== 当前场景：{scene_dir_name} =====")
    scene_report_dir = os.path.join(REPORT_ROOT, scene_dir_name)
    os.makedirs(scene_report_dir, exist_ok=True)

    for img_filename in os.listdir(scene_path):
        if img_filename.lower().endswith(".png"):
            total_img += 1
            img_full_path = os.path.join(scene_path, img_filename)
            print(f"正在处理图片：{img_full_path}")
            inspect_result = single_scene_inspect(img_full_path, scene_dir_name)

            has_any_risk = any(item["has_risk"] for item in inspect_result)
            if has_any_risk:
                risk_img += 1
            if inspect_result[0]["risk_type"] in ("输出JSON格式异常", "处理异常"):
                error_img += 1

            # 缓存本条图片的场景+隐患数据
            all_scene_risk_data.append({
                "scene": scene_dir_name,
                "image_name": img_filename,
                "risk_items": inspect_result
            })

            # 保存单图JSON报告
            report_name = os.path.splitext(img_filename)[0] + ".json"
            report_save_path = os.path.join(scene_report_dir, report_name)
            with open(report_save_path, "w", encoding="utf-8") as f:
                json.dump(inspect_result, f, ensure_ascii=False, indent=2)
            print(f" 报告已保存至：{report_save_path}\n")

# 全局打印基础统计
print("==================== 全部场景图片巡检完成 ====================")
print(f"全部场景总处理图片数量：{total_img}")
print(f"推理异常图片总数量：{error_img}")
print(f"存在安全隐患图片总数量：{risk_img}")
print(f"所有结构化JSON报告统一存放根路径：{REPORT_ROOT}")

# 将全部隐患数据喂给VLM，AI自主生成完整巡检总报告
if total_img > 0:
    print("\n===== 开始调用VLM，基于全部隐患数据生成综合巡检报告 =====")
    # 1. 把缓存的隐患数据转为格式化文本
    data_text = json.dumps(all_scene_risk_data, ensure_ascii=False, indent=1)
    # 2. 报告生成专用Prompt，深度分析，不止复述JSON
    report_prompt = f"""
    
你是资深室内安全巡检工程师，具备消防、居家安全专业资质。现有本次全区域图片检测原始隐患JSON数据，**禁止单纯罗列原始JSON内容**，所有数据仅作为分析依据，撰写一份标准企业级正式安全巡检汇总报告，输出纯Markdown正文，禁止输出JSON、代码块、无关解释。

# 基础巡检统计信息
总检测图片：{total_img} 张
推理异常图片：{error_img} 张
检出安全隐患图片：{risk_img} 张

# 原始隐患数据（仅用于分析，不要大段复制展示）
{data_text}

# 报告强制完整章节结构，缺一不可
## 一、巡检总体概况
1. 巡检覆盖范围：浴室、卧室、走廊（消防通道）、厨房四大核心区域；
2. 图片检测总量、异常样本说明、隐患检出整体概况；
3. 综合安全评级（四档：安全优秀/基本合格/存在较多隐患/重大安全风险），写明评级判定依据。

## 二、全局风险量化研判
1. 统计高、中、低三类风险总条数，计算各类风险占全部隐患的比例；
2. 分别阐述每一类风险可能引发的人身伤害、财产损失事故后果；
3. 单独梳理所有复合叠加风险，分析多重隐患共存时安全危害放大效应。

## 三、全场景共性隐患根源分析
1. 提炼四大场景重复出现的同类隐患，总结背后管理、收纳、用火、通道维护等根源问题；
2. 区分人为习惯问题、硬件缺失问题、日常管理疏漏三类诱因。

## 四、分场景专项隐患深度分析
分浴室、卧室、走廊、厨房逐个分析：
1. 该场景隐患数量、主要风险类型；
2. 高风险点位重点标注，说明该区域专属安全管控重点；
3. 本场景特有风险带来的专属安全威胁。

## 五、分级整改处置清单（核心重点）
按优先级分为三档，每条包含：隐患描述、潜在事故危害、整改时限、落地整改标准、验收要求
1. 一级（立即整改，24小时内完成）：全部高风险隐患；
2. 二级（限期整改，3日内完成）：全部中风险隐患；
3. 三级（常态化维护）：全部低风险隐患。

## 六、分场景标准化长效治理方案
针对浴室防滑管理、卧室杂物收纳、走廊消防通道管控、厨房防火防爆分别给出常态化管理细则，不止简短建议，形成可落地的日常规范。

## 七、异常图片处理方案
针对推理失败、JSON解析异常的图片，说明复测要求、拍摄规范、二次检测流程。

## 八、整体巡检结论与长期安全管理制度建议
1. 本次巡检整体风险总结；
2. 建立定期巡检机制、区域物品收纳规范、安全责任人制度、月度隐患复盘制度；
3. 前置预防措施，从源头减少同类隐患重复出现。

# 严格写作约束
1. 行文正式专业，符合安全巡检公文风格，逻辑层层递进；
2. 禁止大段复制原始JSON数据，仅提炼关键信息用于分析；
3. 重点加粗标注火灾、通道堵塞、重度滑倒、尖锐划伤等高致命风险；
4. 全部整改方案、管理制度具备实操性，不写空泛套话；
5. 仅输出Markdown报告正文，不添加任何对话、注释、JSON代码。
"""
    # 3. 构造纯文本对话消息（无图片）
    messages = [{"role": "user", "content": [{"type": "text", "text": report_prompt}]}]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat_text, return_tensors="pt").to("cuda")

    # 4. 提升生成上限，优化参数适配长文专业报告
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4000,
            temperature=0.35,
            top_p=0.4,
            repetition_penalty=1.08
        )
    raw_report = processor.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    final_report = clean_report_text(raw_report)

    # 5. 保存VLM自主生成的总报告
    report_save_path = os.path.join(REPORT_ROOT, "VLM深度分析巡检总报告.md")
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f" AI深度研判汇总报告已生成：{report_save_path}")
else:
    print("\n 未检测到任何图片，跳过汇总报告生成")
