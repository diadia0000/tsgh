# VALIS 非剛性變換 (Non-Rigid Registration) 原始碼詳細分析

**版本**: valis-wsi 1.2.0  
**分析日期**: 2026-01-27  
**文件路徑**: `/home/hispadmin/tsgh/.venv/lib/python3.11/site-packages/valis/non_rigid_registrars.py`

---

## 📋 目錄

1. [架構總覽](#1-架構總覽)
2. [類別繼承關係](#2-類別繼承關係)
3. [核心抽象類別詳解](#3-核心抽象類別詳解)
4. [具體實現類別詳解](#4-具體實現類別詳解)
5. [位移場 (Displacement Field) 處理](#5-位移場處理)
6. [warp_tools 關鍵函數](#6-warp_tools-關鍵函數)
7. [演算法流程圖](#7-演算法流程圖)
8. [使用建議與注意事項](#8-使用建議與注意事項)

---

## 1. 架構總覽

VALIS 的非剛性變換模組採用**策略模式 (Strategy Pattern)**，提供三種主要的非剛性配準實現：

| 類別 | 底層技術 | 特點 |
|------|----------|------|
| `OpticalFlowWarper` | OpenCV DeepFlow | 最快，預設選項 |
| `RAFTWarper` | PyTorch RAFT | GPU 加速，深度學習 |
| `SimpleElastixWarper` | SimpleITK/Elastix | 最精確，支援控制點 |

### 核心概念：**反向變換 (Backward Transformation)**

```
VALIS 使用 backward transformation:
- 對於輸出圖像中的每個像素 (x', y')
- 查找它在原始圖像中的對應位置 (x, y)
- dx = x - x', dy = y - y'
- bk_dxdy = [dx, dy] 儲存為 (2, H, W) 的陣列
```

---

## 2. 類別繼承關係

```
NonRigidRegistrar (抽象基類)
    │
    ├── NonRigidRegistrarXY (支援控制點)
    │       └── SimpleElastixWarper
    │
    ├── OpticalFlowWarper
    │
    ├── RAFTWarper
    │
    └── NonRigidRegistrarGroupwise (群組配準)
            └── SimpleElastixGroupwiseWarper

NonRigidTileRegistrar (獨立類別，用於大圖像分塊處理)
```

---

## 3. 核心抽象類別詳解

### 3.1 NonRigidRegistrar (L45-336)

這是所有非剛性配準器的抽象基類。

```python
class NonRigidRegistrar(object):
    """Abstract class for non-rigid registration using displacement fields
    
    Warps moving_img to align with fixed_img using backwards transformations
    """
    
    def __init__(self, params=None, rgb=False):
        """
        參數說明：
        ----------
        params : dict
            傳遞給 calc() 的參數字典
            對於 SimpleITK，應為 SimpleITK.ParameterMap
            
        rgb : bool
            是否處理 RGB 圖像
        """
        # 核心屬性初始化
        self.params = params
        self.rgb = rgb
        self.moving_img = None      # 待配準圖像
        self.fixed_img = None       # 參考圖像
        self.mask = None            # 前景遮罩
        self.shape = None           # 圖像形狀 (H, W)
        self.grid_spacing = None    # 變形網格間距
        self.method = None          # 配準方法名稱
        self.warped_image = None    # 配準後圖像
        self.deformation_field_img = None  # 變形場可視化
        self.backward_dx = None     # X 方向位移
        self.backward_dy = None     # Y 方向位移
```

#### 3.1.1 register() 方法 - 核心配準流程 (L189-281)

```python
def register(self, moving_img, fixed_img, mask=None, **kwargs):
    """
    主配準流程，包含以下步驟：
    1. 形狀驗證
    2. 遮罩處理
    3. 呼叫 calc() 計算位移場
    4. 應用位移場
    
    關鍵實現細節：
    """
    
    # [L228-232] 步驟1: 形狀一致性檢查
    moving_shape = warp_tools.get_shape(moving_img)[0:2]
    fixed_shape = warp_tools.get_shape(fixed_img)[0:2]
    assert np.all(moving_shape == fixed_shape), \
        print("Images have different shapes")
    
    # [L234-241] 步驟2: 初始化屬性
    self.shape = moving_shape
    self.moving_img = moving_img
    self.fixed_img = fixed_img
    if mask is None:
        mask = np.full(self.shape, 255, dtype=np.uint8)  # 全部為前景
    self.mask = mask
    
    # [L243-257] 步驟3: 遮罩裁剪 - 只在前景區域進行配準
    # 這是一個優化策略，減少計算量
    if self.mask is not None:
        _, masked_fixed = self.apply_mask(self.mask)
        masked_moving = self.moving_img.copy()
        
        # 計算遮罩的邊界框 (bounding box)
        mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(self.mask))
        min_c, min_r = mask_bbox[0:2]
        max_c, max_r = mask_bbox[0:2] + mask_bbox[2:]
        
        # 裁剪到邊界框範圍
        mask = self.mask[min_r:max_r, min_c:max_c]
        masked_moving = masked_moving[min_r:max_r, min_c:max_c]
        masked_fixed = masked_fixed[min_r:max_r, min_c:max_c]
    
    # [L259-261] 步驟4: 呼叫子類實現的 calc() 方法
    # ** 這是多態的核心 - 不同的配準器實現不同的 calc() **
    bk_dxdy = self.calc(moving_img=masked_moving,
                        fixed_img=masked_fixed,
                        mask=mask, **kwargs)
    
    # [L263-272] 步驟5: 將裁剪區域的位移場放回原始尺寸
    if mask is not None:
        bk_dx = np.zeros(self.shape)
        bk_dx[min_r:max_r, min_c:max_c] = bk_dxdy[0]  # 填入 dx
        bk_dx[self.mask == 0] = 0  # 遮罩外設為 0
        
        bk_dy = np.zeros(self.shape)
        bk_dy[min_r:max_r, min_c:max_c] = bk_dxdy[1]  # 填入 dy
        bk_dy[self.mask == 0] = 0
        
        bk_dxdy = np.array([bk_dx, bk_dy])  # shape: (2, H, W)
    
    # [L274-281] 步驟6: 應用變形並生成結果
    warped_img, warp_grid = self.get_warped_img_and_grid(bk_dxdy)
    
    self.backward_dx = bk_dxdy[..., 0]
    self.backward_dy = bk_dxdy[..., 1]
    self.deformation_field_img = warp_grid
    self.warped_image = warped_img
    
    return warped_img, warp_grid, bk_dxdy
```

#### 3.1.2 calc() 抽象方法 (L147-175)

```python
def calc(self, moving_img, fixed_img, mask, *args, **kwargs):
    """計算位移場 - 由子類實現
    
    Parameters
    ----------
    moving_img : ndarray
        待配準圖像，形狀 (N, M)
    fixed_img : ndarray
        參考圖像，形狀 (N, M)
    mask : ndarray
        前景遮罩，非零為前景
        
    Returns
    -------
    bk_dxdy : ndarray
        形狀 (2, N, M) 的位移陣列
        bk_dxdy[0] = dx (X方向位移)
        bk_dxdy[1] = dy (Y方向位移)
    """
    bk_dxdy = None
    return bk_dxdy  # 抽象方法，返回 None
```

#### 3.1.3 get_warped_img_and_grid() (L313-336)

```python
def get_warped_img_and_grid(self, bk_dxdy):
    """應用變形到圖像和網格
    
    用於：
    1. 生成配準後的圖像
    2. 生成變形可視化網格
    """
    # 使用 warp_tools 中的函數進行實際變形
    warped_img = warp_tools.warp_img(self.moving_img, bk_dxdy=bk_dxdy)
    
    # 生成規則網格用於可視化變形
    grid_img = self.get_grid_image(grid_spacing=16)
    warp_grid = warp_tools.warp_img(grid_img, bk_dxdy=bk_dxdy)
    
    return warped_img, warp_grid
```

---

### 3.2 NonRigidRegistrarXY (L339-499)

支援使用**對應控制點**輔助配準的抽象類。

```python
class NonRigidRegistrarXY(NonRigidRegistrar):
    """
    除了圖像特徵外，還可以使用手動標註的對應點
    來引導非剛性配準
    
    新增屬性：
    - moving_xy: 移動圖像中的控制點 (N, 2)
    - fixed_xy:  固定圖像中的控制點 (N, 2)
    """
    
    def __init__(self, params=None):
        super().__init__(params=params)
        self.moving_xy = None
        self.fixed_xy = None
    
    def register(self, moving_img, fixed_img, mask=None, 
                 moving_xy=None, fixed_xy=None, **kwargs):
        """
        擴展的 register 方法，增加控制點過濾
        """
        # [L470-473] 過濾掉圖像/遮罩外的控制點
        if moving_xy is not None:
            moving_xy, fixed_xy = self.filter_xy(
                moving_xy, fixed_xy,
                moving_img.shape, mask
            )
        
        # 調用父類的 register
        self.moving_xy = moving_xy
        self.fixed_xy = fixed_xy
        warped_img, warp_grid, bk_dxdy = \
            NonRigidRegistrar.register(self, ...)
        
        return warped_img, warp_grid, bk_dxdy
    
    def filter_xy(self, moving_xy, fixed_xy, img_shape_rc, mask=None):
        """過濾遮罩外的控制點"""
        if mask is None:
            mask = np.full(img_shape_rc, 255, dtype=np.uint8)
        
        # 找出在遮罩內的點索引
        moving_inside_idx = warp_tools.get_inside_mask_idx(moving_xy, mask)
        fixed_inside_idx = warp_tools.get_inside_mask_idx(fixed_xy, mask)
        inside_idx = np.intersect1d(moving_inside_idx, fixed_inside_idx)
        
        return moving_xy[inside_idx, :], fixed_xy[inside_idx, :]
```

---

## 4. 具體實現類別詳解

### 4.1 OpticalFlowWarper (L933-1029) - 預設配準器

基於 OpenCV 的**稠密光流**進行非剛性配準。

```python
class OpticalFlowWarper(NonRigidRegistrar):
    """使用稠密光流進行圖像配準
    
    特點：
    - 快速
    - 可能產生翻折 (folding)
    - 提供多種平滑選項
    """
    
    def __init__(self, 
                 params=None, 
                 optical_flow_obj=cv2.optflow.createOptFlow_DeepFlow(),
                 n_grid_pts=50,           # 網格點數量
                 sigma_ratio=0.005,       # 高斯平滑參數
                 paint_size=5000,         # inpaint 尺寸
                 fold_penalty=1e-6,       # 翻折懲罰
                 smoothing_method=None):  # 平滑方法
        """
        Parameters
        ----------
        optical_flow_obj : object
            OpenCV 光流物件，預設使用 DeepFlow
            其他選項：
            - cv2.optflow.createOptFlow_Farneback()
            - cv2.optflow.createOptFlow_DenseRLOF()
            - cv2.optflow.createOptFlow_SimpleFlow()
            
        smoothing_method : str
            "gauss"     - 高斯平滑
            "inpaint"   - 偵測並修復翻折區域
            "regularize"- 網格正則化 (防止翻折)
            None        - 不做平滑
        """
        super().__init__(params)
        
        self.smoothing_method = smoothing_method
        self.sigma_ratio = sigma_ratio
        self.paint_size = paint_size
        self.fold_penalty = fold_penalty
        self.n_grid_pts = n_grid_pts
        
        self.method = optical_flow_obj.__class__.__name__
        self.optical_flow_obj = optical_flow_obj
```

#### calc() 實現 (L1000-1029)

```python
def calc(self, moving_img, fixed_img, *args, **kwargs):
    """計算光流位移場"""
    
    # [L1001-1006] 某些光流方法需要 RGB 輸入
    if self.method in ['createOptFlow_DenseRLOF', 'createOptFlow_SimpleFlow']:
        if moving_img.ndim == 2:
            moving_img = color.gray2rgb(moving_img)
        if fixed_img.ndim == 2:
            fixed_img = color.gray2rgb(fixed_img)
    
    # [L1008-1009] ** 核心計算 **
    # 注意：輸入順序是 (fixed, moving)
    # 這會產生「從 fixed 到 moving」的光流
    # 即 backward flow，用於 image warping
    backward_flow = self.optical_flow_obj.calc(
        fixed_img, moving_img,
        np.zeros(moving_img.shape[0:2])  # 初始化為零流量
    )
    
    # [L1011] 轉換格式: (H,W,2) -> (2,H,W)
    backward_flow = np.array([backward_flow[..., 0], backward_flow[..., 1]])
    
    # [L1012-1027] 平滑處理
    if self.smoothing_method == "gauss":
        # 高斯平滑
        sigma = self.sigma_ratio * np.max(backward_flow[0].shape)
        smooth_dx = filters.gaussian(backward_flow[0], sigma=sigma)
        smooth_dy = filters.gaussian(backward_flow[1], sigma=sigma)
        backward_flow = np.array([smooth_dx, smooth_dy])
        
    elif self.smoothing_method == "inpaint":
        # 偵測翻折並使用 inpainting 修復
        backward_flow = warp_tools.remove_folds_in_dxdy(
            backward_flow,
            n_grid_pts=self.n_grid_pts,
            paint_size=self.paint_size,
            method=self.smoothing_method
        )
        
    elif self.smoothing_method == "regularize":
        # 網格正則化 (基於 Garanzha et al. 2021 論文)
        backward_flow = warp_tools.untangle(
            backward_flow,
            n_grid_pts=self.n_grid_pts,
            penalty=self.fold_penalty,
            mask=self.mask
        )
    
    return np.array(backward_flow)
```

---

### 4.2 RAFTWarper (L1032-1131) - 深度學習配準器

使用 PyTorch 的 **RAFT (Recurrent All-Pairs Field Transforms)** 模型。

```python
class RAFTWarper(NonRigidRegistrar):
    """使用 RAFT 深度學習模型進行光流估計
    
    優點：
    - 更準確的光流估計
    - GPU 加速
    
    缺點：
    - 需要 GPU
    - 記憶體需求較高
    """
    
    def __init__(self, 
                 weights=Raft_Large_Weights.DEFAULT,
                 transform_method="pad",  # "pad" 或 "resize"
                 device=None,
                 rgb=True,
                 quant_img=True,
                 *args, **kwargs):
        """
        Parameters
        ----------
        weights : Raft_Large_Weights
            預訓練權重
            
        transform_method : str
            "pad"    - 用 0 填充使維度可被 8 整除
            "resize" - 調整尺寸使維度可被 8 整除
        """
        super().__init__(rgb=rgb)
        
        # [L1055-1057] 設備選擇
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.quant_img = quant_img
        self.weights = weights
        self.transform_method = transform_method
        
        # [L1062-1063] 載入預訓練模型
        self.model = raft_large(weights=self.weights, progress=False).to(self.device)
        self.model.eval()  # 設為評估模式
```

#### prep_img_for_raft() - 預處理 (L1065-1106)

```python
def prep_img_for_raft(self, img, dim_div=8, method="pad"):
    """準備 RAFT 所需的輸入格式
    
    RAFT 要求：
    1. 3 通道 RGB
    2. 維度可被 8 整除
    """
    # [L1070-1077] 轉換為 3 通道
    if img.ndim == 2:
        img3d = np.dstack(3*[img])  # 灰度轉 RGB
    else:
        if self.quant_img:
            print("quantizing image for non-rigid registration")
            img3d = preprocessing.quantize_image(img)
        else:
            img3d = img
    
    # [L1079-1083] 轉換為 PyTorch tensor
    torch_img = tv_transforms.ToTensor()(img3d).unsqueeze(0)
    
    h, w = torch_img.shape[-2:]
    h_pad = dim_div - h % dim_div  # 計算需要填充的高度
    w_pad = dim_div - w % dim_div  # 計算需要填充的寬度
    
    # [L1085-1106] 填充或調整尺寸
    if method == "pad":
        lpad = w_pad // 2
        rpad = w_pad - lpad
        tpad = h_pad // 2
        bpad = h_pad - tpad
        
        padding = [lpad, tpad, rpad, bpad]
        transformed_img = tv_transforms.Pad(padding)(torch_img)
        img_transform = padding
    else:
        out_h = h + h_pad
        out_w = w + w_pad
        transformed_img = tv_transforms.Resize(
            size=[out_h, out_w], antialias=False
        )(torch_img)
        img_transform = np.array([out_w/w, out_h/h])  # 縮放因子
    
    return transformed_img, img_transform
```

#### calc() 實現 (L1108-1131)

```python
def calc(self, moving_img, fixed_img, *args, **kwargs):
    """使用 RAFT 計算光流"""
    
    # [L1109-1110] 預處理
    transformed_moving_img, moving_transform = self.prep_img_for_raft(
        moving_img, method=self.transform_method
    )
    transformed_fixed_img, fixed_transform = self.prep_img_for_raft(
        fixed_img, method=self.transform_method
    )
    
    # [L1112] 應用 RAFT 的標準化轉換
    transformed_moving_img, transformed_fixed_img = self.weights.transforms()(
        transformed_moving_img, transformed_fixed_img
    )
    
    # [L1114-1115] ** 核心推理 **
    # 注意順序：(fixed, moving) -> backward flow
    list_of_flows = self.model(
        transformed_fixed_img.to(self.device), 
        transformed_moving_img.to(self.device)
    )
    
    # 取最後一次迭代的結果
    dxdy = list_of_flows[-1].squeeze(0).detach().numpy()
    
    # [L1117-1129] 移除填充或調整尺寸
    if self.transform_method == "pad" and len(moving_transform) == 4:
        lp, tp, rp, bp = moving_transform
        cropped_dx = dxdy[0][tp:dxdy.shape[1]-bp, lp:dxdy.shape[2]-rp]
        cropped_dy = dxdy[1][tp:dxdy.shape[1]-bp, lp:dxdy.shape[2]-rp]
        backward_flow = np.array([cropped_dx, cropped_dy])
    elif len(moving_transform) == 2:
        scaled_dxdy = warp_tools.scale_dxdy(dxdy, out_shape_rc=moving_img.shape[0:2])
        backward_flow = np.array([scaled_dxdy[..., 0], scaled_dxdy[..., 1]])
    
    return backward_flow
```

---

### 4.3 SimpleElastixWarper (L657-930) - 醫學影像配準

基於 **SimpleITK/Elastix** 的 B-spline 非剛性配準。

```python
class SimpleElastixWarper(NonRigidRegistrarXY):
    """使用 SimpleElastix 進行配準
    
    特點：
    - 支援控制點約束
    - B-spline 變形模型
    - 多種度量指標組合
    """
    
    def __init__(self, 
                 params=None, 
                 ammi_weight=0.33,            # 互信息權重
                 bending_penalty_weight=0.33, # 彎曲懲罰權重
                 kp_weight=0.33):             # 控制點權重
        """
        度量指標權重說明：
        - AdvancedMattesMutualInformation: 度量圖像相似度
        - TransformBendingEnergyPenalty: 防止過度變形
        - CorrespondingPointsEuclideanDistanceMetric: 控制點對齊
        """
        super().__init__(params=params)
        
        self.ammi_weight = ammi_weight
        self.bending_penalty_weight = bending_penalty_weight
        self.kp_weight = kp_weight
```

#### get_default_params() - 預設參數 (L687-718)

```python
@staticmethod
def get_default_params(img_shape, grid_spacing_ratio=0.025):
    """獲取 Elastix 的預設配準參數
    
    參考: https://simpleelastix.readthedocs.io/Introduction.html
    """
    p = sitk.GetDefaultParameterMap("bspline")  # B-spline 變形模型
    
    # 度量指標設定
    p["Metric"] = [
        'AdvancedMattesMutualInformation',  # 互信息
        'TransformBendingEnergyPenalty'     # 變形平滑度懲罰
    ]
    
    # 優化器設定
    p["MaximumNumberOfIterations"] = ['1500']
    p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
    
    # 圖像金字塔設定
    p['FixedImagePyramid'] = ["FixedRecursiveImagePyramid"]
    p['MovingImagePyramid'] = ["MovingRecursiveImagePyramid"]
    
    # 採樣策略
    p['Interpolator'] = ["BSplineInterpolator"]
    p["ImageSampler"] = ["RandomCoordinate"]
    p["MetricSamplingStrategy"] = ["None"]  # 使用所有點
    p["UseRandomSampleRegion"] = ["true"]
    p["NumberOfSpatialSamples"] = ["3000"]
    p["NewSamplesEveryIteration"] = ["true"]
    p["SampleRegionSize"] = [str(min([img_shape[1]//3, img_shape[0]//3]))]
    
    # 變形網格設定
    grid_spacing_x = img_shape[1] * grid_spacing_ratio
    grid_spacing_y = img_shape[0] * grid_spacing_ratio
    grid_spacing = str(int(np.mean([grid_spacing_x, grid_spacing_y])))
    p["FinalGridSpacingInPhysicalUnits"] = [grid_spacing]
    
    # 其他設定
    p["ErodeMask"] = ["true"]
    p["NumberOfHistogramBins"] = ["32"]
    p["HowToCombineTransforms"] = ["Compose"]
    p["WriteResultImage"] = ["false"]
    
    return p
```

#### run_elastix() - 執行配準 (L781-897)

```python
def run_elastix(self, moving_img, fixed_img, moving_xy=None, fixed_xy=None,
                params=None, mask=None):
    """執行 SimpleElastix 配準
    
    核心步驟：
    1. 設定控制點（如有）
    2. 執行配準
    3. 提取位移場
    """
    
    # [L813] 創建 Elastix 過濾器
    elastix_image_filter_obj = sitk.ElastixImageFilter()
    
    # [L815-837] 處理控制點
    if moving_xy is not None and fixed_xy is not None:
        # 寫入臨時控制點文件
        rand_id = np.random.randint(0, 10000)
        fixed_kp_fname = os.path.join(
            pathlib.Path(__file__).parent,
            f".{rand_id}_fixedPointSet.pts"
        )
        moving_kp_fname = os.path.join(
            pathlib.Path(__file__).parent,
            f".{rand_id}_.movingPointSet.pts"
        )
        
        self.write_elastix_kp(fixed_xy, fixed_kp_fname)
        self.write_elastix_kp(moving_xy, moving_kp_fname)
        
        # 添加控制點度量
        kp_dist_met = "CorrespondingPointsEuclideanDistanceMetric"
        current_metrics = list(params["Metric"])
        if not self._params_provided or kp_dist_met not in current_metrics:
            current_metrics.append(kp_dist_met)
            params["Metric"] = current_metrics
            weights = np.array([
                self.ammi_weight,
                self.bending_penalty_weight,
                self.kp_weight
            ])
        
        elastix_image_filter_obj.SetFixedPointSetFileName(fixed_kp_fname)
        elastix_image_filter_obj.SetMovingPointSetFileName(moving_kp_fname)
    else:
        weights = np.array([self.ammi_weight, self.bending_penalty_weight])
    
    # [L841-846] 設定度量權重
    weights /= weights.sum()  # 歸一化
    n_metrics = len(params["Metric"])
    n_res = eval(params["NumberOfResolutions"][0])
    for r in range(n_metrics):
        params[f'Metric{r}Weight'] = [str(weights[r])] * n_res
    
    elastix_image_filter_obj.SetParameterMap(params)
    
    # [L850-862] 執行配準
    sitk_moving = sitk.GetImageFromArray(moving_img)
    sitk_fixed = sitk.GetImageFromArray(fixed_img)
    elastix_image_filter_obj.SetMovingImage(sitk_moving)
    elastix_image_filter_obj.SetFixedImage(sitk_fixed)
    
    if mask is not None:
        sitk_mask = sitk.Cast(
            sitk.GetImageFromArray(mask.astype(np.uint8)),
            sitk.sitkUInt8
        )
        elastix_image_filter_obj.SetFixedMask(sitk_mask)
    
    elastix_image_filter_obj.Execute()  # ** 執行配準 **
    
    # [L864-869] 提取位移場
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(
        elastix_image_filter_obj.GetTransformParameterMap()
    )
    transformixImageFilter.ComputeDeformationFieldOn()
    transformixImageFilter.Execute()
    deformationField = sitk.GetArrayFromImage(
        transformixImageFilter.GetDeformationField()
    )
    
    # [L871-880] 獲取配準結果
    resultImage = sitk.GetArrayFromImage(
        elastix_image_filter_obj.GetResultImage()
    )
    
    # 清理臨時文件...
    
    return resultImage, warped_grid, deformationField, \
           elastix_image_filter_obj, transformixImageFilter
```

---

### 4.4 NonRigidTileRegistrar (L1270-1601) - 大圖像分塊配準

專門處理**超大圖像**（如 WSI 全玻片影像）的分塊配準器。

```python
class NonRigidTileRegistrar(object):
    """分塊非剛性配準
    
    策略：
    1. 將大圖像分割為小塊
    2. 對每個小塊進行配準
    3. 將位移場拼接回完整尺寸
    
    適用場景：
    - 超大 WSI 圖像
    - 記憶體有限的環境
    """
    
    def __init__(self, params=None, tile_wh=512, tile_buffer=100):
        """
        Parameters
        ----------
        tile_wh : int
            分塊的寬度和高度 (像素)
            
        tile_buffer : int
            相鄰分塊間的重疊量，用於平滑拼接
        """
        self.tile_wh = tile_wh
        self.tile_buffer = tile_buffer
        self.params = params
        
        # pyvips 用於處理大圖像
        self.moving_img = None  # pyvips.Image
        self.fixed_img = None   # pyvips.Image
        self.mask = None
        
        # 每個分塊的位移場
        self.bk_dxdy_tiles = None  # 反向位移
        self.fwd_dxdy_tiles = None # 正向位移
```

#### reg_tile() - 單塊配準 (L1385-1466)

```python
def reg_tile(self, tile_idx, lock):
    """配準單個分塊"""
    
    with lock:  # 使用鎖保護圖像訪問
        # [L1389-1394] 提取分塊
        tile_bbox_xywh = self.expanded_bboxes[tile_idx]
        moving_tile = self.moving_img.extract_area(*tile_bbox_xywh)
        fixed_tile = self.fixed_img.extract_area(*tile_bbox_xywh)
        
        np_fixed = warp_tools.vips2numpy(fixed_tile)
        np_moving = warp_tools.vips2numpy(moving_tile)
        
        # [L1396-1400] 提取遮罩
        if self.mask is not None:
            tile_mask = self.mask.extract_area(*tile_bbox_xywh)
            np_mask = warp_tools.vips2numpy(tile_mask)
        else:
            np_mask = None
        
        # [L1412-1423] 檢查是否為空白區域
        is_empty = (fixed_tile.max() == fixed_tile.min() or 
                    moving_tile.max() == moving_tile.min())
        if np_mask is not None:
            is_empty = is_empty or np_mask.max() == 0
        
        if is_empty:
            # 空白區域返回零位移
            empty_dxdy = pyvips.Image.black(
                moving_tile.width, moving_tile.height, bands=2
            ).cast("float")
            self.bk_dxdy_tiles[tile_idx] = empty_dxdy
            self.fwd_dxdy_tiles[tile_idx] = empty_dxdy
            return None
        
        # [L1425-1449] 預處理分塊
        if self.moving_processer_cls is not None:
            moving_processed = self.process_tile(...)
        else:
            if np_moving.ndim > 2:
                moving_g = np.abs(1 - skcolor.rgb2gray(np_moving))
                moving_processed = util.img_as_ubyte(moving_g)
            else:
                moving_processed = np_moving
        
        # 對 fixed 做類似處理...
        
        # [L1451] 標準化
        moving_normed, fixed_normed = self.norm_tiles(
            moving_processed, fixed_processed, np_mask
        )
        
        # [L1453-1459] 執行配準
        if inspect.isclass(self.non_rigid_registrar_cls):
            tile_non_rigid_reg_obj = self.non_rigid_registrar_cls()
        else:
            tile_non_rigid_reg_obj = self.non_rigid_registrar_cls
        
        _, _, bk_dxdy = tile_non_rigid_reg_obj.register(
            moving_normed, fixed_normed
        )
        
        # [L1460] 計算正向位移（用於點變換）
        fwd_dxdy = warp_tools.get_inverse_field(bk_dxdy)
        
        # [L1462-1466] 儲存結果
        vips_tile_bk_dxdy = warp_tools.numpy2vips(
            np.dstack(bk_dxdy).astype(np.float32)
        )
        vips_tile_fwd_dxdy = warp_tools.numpy2vips(
            np.dstack(fwd_dxdy).astype(np.float32)
        )
        
        self.bk_dxdy_tiles[tile_idx] = vips_tile_bk_dxdy
        self.fwd_dxdy_tiles[tile_idx] = vips_tile_fwd_dxdy
```

#### calc() - 並行配準 (L1468-1484)

```python
def calc(self, *args, **kwargs):
    """並行計算所有分塊的位移場"""
    
    print("======== Registering tiles\n")
    
    n_cpu = valtils.get_ncpus_available() - 1  # 使用 n-1 個 CPU
    
    # 使用多執行緒並行處理
    lock = multiprocessing.Lock()
    args = [{"tile_idx": i, "lock": lock} for i in range(self.n_tiles)]
    res = pqdm(args, self.reg_tile, n_jobs=n_cpu, 
               unit="image", leave=None, argument_type='kwargs')
    
    # 拼接分塊
    bk_dxdy = warp_tools.stitch_tiles(
        self.bk_dxdy_tiles, self.expanded_bboxes, 
        self.n_rows, self.n_cols, self.tile_buffer
    )
    fwd_dxdy = warp_tools.stitch_tiles(
        self.fwd_dxdy_tiles, self.expanded_bboxes,
        self.n_rows, self.n_cols, self.tile_buffer
    )
    
    return bk_dxdy, fwd_dxdy
```

---

## 5. 位移場處理

### 5.1 位移場格式

```python
# 位移場格式: (2, H, W)
bk_dxdy[0]  # dx: X 方向位移 (列方向)
bk_dxdy[1]  # dy: Y 方向位移 (行方向)

# 含義：對於輸出位置 (x', y')
# 其對應的輸入位置為:
# x = x' + dx[y', x']
# y = y' + dy[y', x']
```

### 5.2 正向與反向位移

```python
# warp_tools.py L1928-1938
def get_inverse_field(backwards_xy_deltas, n_inter=10):
    """將反向位移場轉換為正向位移場
    
    使用 SimpleITK 的迭代逆變換方法
    
    用途：
    - backward -> 用於圖像變形 (image warping)
    - forward  -> 用於點變換 (point warping)
    """
    sitk_bk_dxdy = sitk.GetImageFromArray(
        np.dstack(backwards_xy_deltas), isVector=True
    )
    sitk_fw_dxdy = sitk.IterativeInverseDisplacementField(
        sitk_bk_dxdy, numberOfIterations=n_inter
    )
    fwd_dxdy = sitk.GetArrayFromImage(sitk_fw_dxdy)
    fwd_dxdy = [fwd_dxdy[..., 0], fwd_dxdy[..., 1]]
    
    return fwd_dxdy
```

---

## 6. warp_tools 關鍵函數

### 6.1 warp_img() - 圖像變形 (L1021-1230)

```python
def warp_img(img, M=None, bk_dxdy=None, out_shape_rc=None,
             transformation_src_shape_rc=None,
             transformation_dst_shape_rc=None,
             bbox_xywh=None,
             bg_color=None,
             interp_method="bicubic"):
    """使用剛性和/或非剛性變換來變形圖像
    
    核心流程：
    1. 確定縮放因子
    2. 應用剛性變換 (affine)
    3. 應用非剛性變換 (displacement field)
    4. 可選裁剪
    """
    
    # [L1078-1081] 處理 numpy/pyvips 格式
    is_array = False
    if not isinstance(img, pyvips.Image):
        is_array = True
        img = numpy2vips(img)
    
    # [L1148-1178] ** 剛性變換 **
    if do_rigid:
        # 縮放變換矩陣
        if not np.all(src_sxy == 1):
            img_corners_xy = get_corners_of_image(src_shape_rc)[:, ::-1]
            warped_corners = warp_xy(img_corners_xy, M=M, ...)
            M_tform = transform.ProjectiveTransform()
            M_tform.estimate(warped_corners, img_corners_xy)
            warp_M = M_tform.params
        else:
            warp_M = M
        
        tx, ty = warp_M[:2, 2]
        warp_M = np.linalg.inv(warp_M)  # 反轉用於 backward warping
        vips_M = warp_M[:2, :2].reshape(-1).tolist()
        
        # 使用 pyvips 的 affine 方法
        affine_warped = img.affine(vips_M,
            oarea=[0, 0, out_shape_rc[1], out_shape_rc[0]],
            interpolate=interpolator,
            idx=-tx, idy=-ty,
            premultiplied=True,
            background=bg_color,
            extend=bg_extender
        )
    else:
        affine_warped = img
    
    # [L1180-1222] ** 非剛性變換 **
    if do_non_rigid:
        # 準備位移場
        if not isinstance(bk_dxdy, pyvips.Image):
            temp_dxdy = numpy2vips(np.dstack(bk_dxdy))
        else:
            temp_dxdy = bk_dxdy
        
        # 縮放位移場
        if dst_sxy is not None:
            scaled_dx = float(dst_sxy[0]) * temp_dxdy[0]
            scaled_dy = float(dst_sxy[1]) * temp_dxdy[1]
            vips_dxdy = scaled_dx.bandjoin(scaled_dy)
        else:
            vips_dxdy = temp_dxdy
        
        # 調整位移場尺寸
        warp_dxdy = vips_dxdy.affine(S,
            oarea=[0, 0, out_shape_rc[1], out_shape_rc[0]],
            interpolate=interpolator,
            premultiplied=True
        )
        
        # ** 核心：創建索引映射 **
        index = pyvips.Image.xyz(affine_warped.width, affine_warped.height)
        warp_index = (index[0] + warp_dxdy[0]).bandjoin(index[1] + warp_dxdy[1])
        
        # ** 應用映射進行變形 **
        warped = affine_warped.mapim(warp_index,
            premultiplied=True,
            background=bg_color,
            extend=bg_extender,
            interpolate=interpolator
        )
    
    return warped
```

### 6.2 warp_xy_non_rigid() - 點變形 (L1969-1992)

```python
def warp_xy_non_rigid(xy, dxdy, displacement_shape_rc=None):
    """使用位移場變形點座標
    
    使用雙變量樣條插值來獲取任意位置的位移
    """
    single_pt = xy.ndim == 1
    if single_pt:
        xy = np.array([xy])
    
    if displacement_shape_rc is None:
        displacement_shape_rc = dxdy[0].shape
    
    bbox = [0, displacement_shape_rc[0], 0, displacement_shape_rc[1]]
    grid_r = np.arange(displacement_shape_rc[0])
    grid_c = np.arange(displacement_shape_rc[1])
    
    # 創建插值器
    interp_dx = RectBivariateSpline(grid_r, grid_c, dxdy[0], bbox=bbox)
    interp_dy = RectBivariateSpline(grid_r, grid_c, dxdy[1], bbox=bbox)
    
    # 計算新位置
    nr_x = xy[:, 0] + interp_dx(xy[:, 1], xy[:, 0], grid=False)
    nr_y = xy[:, 1] + interp_dy(xy[:, 1], xy[:, 0], grid=False)
    
    nr_xy = np.dstack([nr_x, nr_y])[0]
    
    return nr_xy
```

---

## 7. 演算法流程圖

```
┌─────────────────────────────────────────────────────────────────┐
│                      VALIS 非剛性配準流程                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. 輸入圖像                                                     │
│     ├── moving_img: 待配準圖像                                   │
│     ├── fixed_img:  參考圖像                                     │
│     └── mask:       前景遮罩 (可選)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. 形狀驗證 & 遮罩處理                                          │
│     ├── 確認 moving 和 fixed 形狀一致                            │
│     ├── 計算遮罩邊界框                                           │
│     └── 裁剪到前景區域 (減少計算量)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. 計算位移場 (calc 方法)                                       │
│     ┌────────────────────────────────────────────────────────┐  │
│     │  OpticalFlowWarper                                      │  │
│     │  └── cv2.optflow.calc(fixed, moving)                   │  │
│     │      └── 可選平滑: gauss/inpaint/regularize            │  │
│     ├────────────────────────────────────────────────────────┤  │
│     │  RAFTWarper                                             │  │
│     │  └── RAFT model inference                              │  │
│     │      └── 處理填充/縮放                                   │  │
│     ├────────────────────────────────────────────────────────┤  │
│     │  SimpleElastixWarper                                    │  │
│     │  └── sitk.ElastixImageFilter                           │  │
│     │      └── B-spline 變形 + 控制點約束                      │  │
│     └────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│     輸出: bk_dxdy (2, H, W) - 反向位移場                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. 填充回原始尺寸                                               │
│     ├── 將裁剪區域的位移場放回全尺寸陣列                          │
│     └── 遮罩外區域設為零位移                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. 應用變形                                                     │
│     ├── warp_tools.warp_img(moving_img, bk_dxdy)                │
│     │   └── pyvips.Image.mapim()                                │
│     └── 生成變形網格可視化                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. 輸出                                                         │
│     ├── warped_img:   配準後圖像                                 │
│     ├── warp_grid:    變形網格 (可視化用)                         │
│     └── bk_dxdy:      位移場 (可用於其他變換)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 使用建議與注意事項

### 8.1 選擇配準器

| 場景 | 推薦配準器 | 理由 |
|------|-----------|------|
| 一般快速配準 | `OpticalFlowWarper` | 預設選項，速度最快 |
| 需要高精度 | `SimpleElastixWarper` | B-spline 模型更靈活 |
| 有 GPU 且需要高精度 | `RAFTWarper` | 深度學習模型效果好 |
| 超大 WSI 圖像 | `NonRigidTileRegistrar` | 分塊處理節省記憶體 |
| 需要手動控制點 | `SimpleElastixWarper` | 唯一支援控制點的配準器 |

### 8.2 重要參數調優

```python
# OpticalFlowWarper
OpticalFlowWarper(
    smoothing_method="regularize",  # 推薦：防止翻折
    fold_penalty=1e-6,              # 調高可減少變形程度
    n_grid_pts=50                   # 網格密度
)

# SimpleElastixWarper
SimpleElastixWarper(
    ammi_weight=0.5,               # 增加圖像相似性權重
    bending_penalty_weight=0.3,    # 增加可減少過度變形
    kp_weight=0.2                  # 控制點權重
)

# NonRigidTileRegistrar
NonRigidTileRegistrar(
    tile_wh=512,                   # 分塊大小
    tile_buffer=100                # 重疊量，建議 >= 50
)
```

### 8.3 注意事項

1. **位移場方向**
   ```python
   # VALIS 使用 backward transformation
   # bk_dxdy 表示: 輸出位置 -> 輸入位置 的映射
   # 要變換點座標，需要使用 forward displacement (fwd_dxdy)
   fwd_dxdy = warp_tools.get_inverse_field(bk_dxdy)
   ```

2. **記憶體管理**
   ```python
   # 對於大圖像，使用 pyvips 而非 numpy
   # NonRigidTileRegistrar 專為此設計
   ```

3. **遮罩處理**
   ```python
   # 提供遮罩可以：
   # 1. 減少計算量 (只在前景區域配準)
   # 2. 避免背景干擾配準結果
   ```

4. **光流翻折問題**
   ```python
   # OpticalFlowWarper 可能產生翻折 (folding)
   # 使用 smoothing_method="regularize" 可以緩解
   # 或改用 SimpleElastixWarper (B-spline 天然無翻折)
   ```

---

## 附錄：關鍵依賴

```
- OpenCV (cv2.optflow): 光流計算
- PyTorch (torchvision.models.optical_flow): RAFT 模型
- SimpleITK (sitk): Elastix 配準
- pyvips: 大圖像處理
- scipy: 插值、優化
- scikit-image: 圖像變換
```
