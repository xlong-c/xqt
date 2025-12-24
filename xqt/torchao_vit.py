import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import timm
from torchvision.datasets import ImageFolder  # 替换 ImageNet 为 ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader, Subset
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
import time

def get_autocast():
    # Ada架构(40系显卡)对bfloat16支持更好
    return torch.autocast(device_type='cuda', dtype=torch.bfloat16)

# -------------------------- 全局配置 --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "FP8量化建议在支持的NVIDIA GPU上运行"

BATCH_SIZE = 64  # 增大 Batch Size 以体现 FP8 优势
IMAGE_NET_ROOT = r"E:\dataset\imagenet"  # 已修正，类别文件夹直接在此目录下
RESULT_CSV = "layer_error_analysis.csv"  # 误差报告保存路径

# 数据预处理(与ViT预训练一致)
def get_transform():
    return Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# -------------------------- 1. 加载原始模型和量化模型 --------------------------
def load_models():
    """加载原始FP32模型和FP8量化模型(针对4070Ti Super优化)"""
    model_name = 'vit_small_patch16_224'
    # 1. 加载原始模型 (使用 BF16 以提高效率并匹配量化算子)
    model_fp32 = timm.create_model(model_name, pretrained=True).to(DEVICE).to(torch.bfloat16)
    model_fp32.eval()
    
    # 2. 加载并量化模型(FP8 权重+激活动态量化)
    model_quant = timm.create_model(model_name, pretrained=True).to(DEVICE).to(torch.bfloat16)
    model_quant.eval()
    
    # 定义量化策略: 权重和激活都使用FP8
    quant_strategy = Float8DynamicActivationFloat8WeightConfig()
    
    print(f"正在为 {model_name} 执行 FP8 量化...")
    
    # 细化过滤逻辑：仅量化 Linear 层且排除 head
    def filter_fn(module, name):
        return isinstance(module, nn.Linear) and "head" not in name

    quantize_(model_quant, quant_strategy, filter_fn)

    return model_fp32, model_quant

# -------------------------- 2. 计算误差的核心函数 --------------------------
def calculate_error(a, b):
    """计算两个张量的误差指标:MAE/MSE/余弦相似度"""
    # 转为 float32 (numpy 不支持 bfloat16)
    a_np = a.detach().to(torch.float32).flatten().cpu().numpy()
    b_np = b.detach().to(torch.float32).flatten().cpu().numpy()
    
    # 计算指标
    mae = np.mean(np.abs(a_np - b_np))  # 平均绝对误差
    mse = np.mean((a_np - b_np) **2)    # 均方误差
    # 余弦相似度(避免除零)
    norm_a = np.linalg.norm(a_np) + 1e-8
    norm_b = np.linalg.norm(b_np) + 1e-8
    cos_sim = np.dot(a_np, b_np) / (norm_a * norm_b)
    
    return {
        "mae": round(float(mae), 6),
        "mse": round(float(mse), 6),
        "cos_sim": round(float(cos_sim), 6)
    }

def get_layer_weights(model, layer_name):
    """根据层名提取权重(处理量化张量的反量化)"""
    # 遍历模型找到目标层
    for name, module in model.named_modules():
        if name == layer_name:
            # 提取权重(优先取weight,无则取bias)
            if hasattr(module, "weight") and module.weight is not None:
                weight = module.weight
                # 量化权重反量化为FP32
                if hasattr(weight, "dequantize"):
                    weight = weight.dequantize()
                return weight.detach()
            else:
                return None
    return None

def benchmark_performance(model, model_name="Model", num_iters=100):
    """测试模型推理性能"""
    test_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE)
    # 预热
    print(f"正在预热 {model_name}...")
    with torch.no_grad(), get_autocast():
        for _ in range(10):
            model(test_input)
    
    # 同步GPU
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad(), get_autocast():
        for _ in range(num_iters):
            model(test_input)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iters * 1000
    print(f"{model_name} 平均推理时间: {avg_time:.2f} ms / batch (Batch size: {BATCH_SIZE})")
    return avg_time

# -------------------------- 3. 逐层误差分析 --------------------------
def analyze_layer_errors(model_fp32, model_quant):
    """逐层分析权重误差和激活误差"""
    # 1. 准备测试数据(用真实数据计算激活误差)
    # 使用 ImageFolder 加载解压后的数据
    test_dataset = Subset(ImageFolder(IMAGE_NET_ROOT, transform=get_transform()), 
                          range(BATCH_SIZE))  # 取1个批次即可
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_input, _ = next(iter(test_loader))
    test_input = test_input.to(DEVICE, dtype=torch.bfloat16)
    
    # 2. 注册钩子,提取所有层的激活输出
    activations_fp32 = {}  # 原始模型激活
    activations_quant = {}  # 量化模型激活
    
    # 注册钩子函数
    def hook_fn(name):
        def fn(module, input, output):
            # 处理tuple输出(如Attention层)
            if isinstance(output, tuple):
                output = output[0]
            activations_fp32[name] = output.detach()
        return fn
    
    def hook_fn_quant(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations_quant[name] = output.detach()
        return fn
    
    # 为原始模型注册钩子
    hooks_fp32 = []
    for name, module in model_fp32.named_modules():
        # 仅分析核心层(跳过无关层如Dropout/Identity)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention, nn.LayerNorm)):
            hooks_fp32.append(module.register_forward_hook(hook_fn(name)))
    
    # 为量化模型注册钩子
    hooks_quant = []
    for name, module in model_quant.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention, nn.LayerNorm)):
            hooks_quant.append(module.register_forward_hook(hook_fn_quant(name)))
    
    # 前向传播,提取激活
    with torch.no_grad(), get_autocast():
        model_fp32(test_input)
        model_quant(test_input)
    
    # 移除钩子
    for h in hooks_fp32:
        h.remove()
    for h in hooks_quant:
        h.remove()
    
    # 3. 逐层计算误差
    layer_error_list = []
    print("\n开始逐层误差分析...")
    for layer_name in tqdm(activations_fp32.keys(), desc="分析进度"):
        # 跳过量化模型中无激活的层
        if layer_name not in activations_quant:
            continue
        
        # ---------- 计算激活误差 ----------
        act_fp32 = activations_fp32[layer_name]
        act_quant = activations_quant[layer_name]
        # 确保形状一致(处理可能的维度差异)
        if act_fp32.shape != act_quant.shape:
            print(f"警告:{layer_name} 激活形状不一致,跳过")
            continue
        act_error = calculate_error(act_fp32, act_quant)
        
        # ---------- 计算权重误差 ----------
        weight_fp32 = get_layer_weights(model_fp32, layer_name)
        weight_quant = get_layer_weights(model_quant, layer_name)
        weight_error = {
            "mae": None,
            "mse": None,
            "cos_sim": None
        }
        if weight_fp32 is not None and weight_quant is not None:
            if weight_fp32.shape == weight_quant.shape:
                weight_error = calculate_error(weight_fp32, weight_quant)
            else:
                print(f"警告:{layer_name} 权重形状不一致")
        
        # ---------- 记录结果 ----------
        layer_error_list.append({
            "layer_name": layer_name,
            "layer_type": str(type(model_fp32.get_submodule(layer_name)).__name__),
            # 激活误差
            "act_mae": act_error["mae"],
            "act_mse": act_error["mse"],
            "act_cos_sim": act_error["cos_sim"],
            # 权重误差
            "weight_mae": weight_error["mae"],
            "weight_mse": weight_error["mse"],
            "weight_cos_sim": weight_error["cos_sim"],
            # 误差排序关键指标(激活MAE)
            "sort_key": act_error["mae"]
        })
    
    # 4. 按激活MAE降序排序(高误差在前)
    layer_error_list = sorted(layer_error_list, key=lambda x: x["sort_key"], reverse=True)
    
    # 5. 保存结果到CSV
    df = pd.DataFrame(layer_error_list)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8")
    print(f"\n误差报告已保存至:{RESULT_CSV}")
    
    # 6. 打印TOP10高误差层
    print("\n==================== TOP10 高误差层(按激活MAE排序) ====================")
    print(f"{'排名':<4} {'层名':<50} {'层类型':<20} {'激活MAE':<10} {'权重MAE':<10} {'激活余弦相似度':<15}")
    print("-" * 120)
    for i, layer in enumerate(layer_error_list[:10]):
        print(f"{i+1:<4} {layer['layer_name']:<50} {layer['layer_type']:<20} {layer['act_mae']:<10} {layer['weight_mae']:<10} {layer['act_cos_sim']:<15}")
    
    return layer_error_list

# -------------------------- 主程序执行 --------------------------
if __name__ == "__main__":
    # 加载模型
    print("加载模型中...")
    model_fp32, model_quant = load_models()
    
    # 逐层误差分析 (在编译前进行，确保钩子生效)
    print("\n==================== 逐层误差分析 ====================")
    layer_errors = analyze_layer_errors(model_fp32, model_quant)
    
    # 性能测试
    print("\n==================== 性能基准测试 (4070Ti Super) ====================")
    print("正在预热并编译模型 (这可能需要几分钟)...")
    
    # 编译量化模型以获得加速
    model_quant_compiled = torch.compile(model_quant)
    
    # 测试 BF16 性能
    time_fp32 = benchmark_performance(model_fp32, "FP32 (BF16)")
    
    # 测试 Compiled FP8 性能
    time_fp8 = benchmark_performance(model_quant_compiled, "FP8 (Compiled)")
    
    print(f"\n加速比 (Compiled FP8 vs BF16): {time_fp32 / time_fp8:.2f}x")
    
    # 额外提示:高误差层优化建议
    print("\n==================== 高误差层优化建议 ====================")
    print("1. 分类头/输出层(head):跳过该层量化(保留FP16/BF16);")
    print("2. 激活值敏感层:尝试使用float8_dynamic_activation_float8_weight或float8_static;")
    print("3. LayerNorm层:跳过量化(LN对低比特敏感);")
    print("4. 权重缩放:如果per-tensor效果不佳, 尝试开启per-channel缩放;")
    print("5. 所有层:增加校准数据量(≥100批次),使用真实场景数据校准。")