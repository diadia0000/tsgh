import torch

# 載入 checkpoint
checkpoint = torch.load("/home/sec312/tsgh/unet_mask/models/densenet121/checkpoint_epoch092_val_miou_0.9289.pth", weights_only=False)

print("--- Checkpoint 結構 ---")
print(f"Checkpoint 類型: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print(f"頂層 keys: {list(checkpoint.keys())}")
    
    # 嘗試找到 model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("\n使用 'model_state_dict' 作為權重來源")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("\n使用 'state_dict' 作為權重來源")
    else:
        state_dict = checkpoint
        print("\n直接使用 checkpoint 作為權重來源")
else:
    state_dict = checkpoint

print(f"\n總共有 {len(state_dict)} 個 keys")
print(f"前 10 個 keys: {list(state_dict.keys())[:10]}")

print("\n--- 模型權重概況 ---")
weight_count = 0
for layer_name, weights in state_dict.items():
    if "weight" in layer_name and hasattr(weights, 'mean'):
        weight_count += 1
        if weight_count <= 10:  # 只顯示前 10 個權重層
            print(f"層名: {layer_name}")
            print(f"  形狀: {weights.shape}")
            print(f"  平均值: {weights.mean().item():.6f}")
            print(f"  標準差: {weights.std().item():.6f}")

print(f"\n總共有 {weight_count} 個 weight 層")