"""ROI（分析範圍）在批次驅動端的判斷。

ROI 的知識散在三層：``backend/schemas/hybrid.py`` 驗證前端送來的四個數字、
``m0_module/m0_reader.py`` 的 ``PrecutStream`` 驗證它是否落在切片內並把格線平移到
ROI 原點、而 ``hybrid_pipeline.run_batch`` 只需要知道**格線從哪個座標開始**，好把
原點交給 ``compute_tile_geometry`` 做「第一格在不在」的驗證。

本模組只承擔最後那一項，讓 ``run_batch`` 不必自己認得 ``PrecutStream.region``
這個屬性的形狀。
"""
from typing import Optional, Tuple


def roi_origin(tile_stream: Optional[object]) -> Tuple[int, int]:
    """回傳格線原點 ``(x, y)``；整片模式為 ``(0, 0)``。

    ``tile_stream`` 是 ``run_batch`` 收到的可選 tile 供給者。它被宣告為
    ``Optional[object]``（不是 ``PrecutStream``）——掃目錄模式根本不傳、測試會傳自製
    的假 stream——所以這裡用 ``getattr`` 取 ``region`` 而非直接存取：沒有 ``region``
    屬性的供給者一律視為整片。

    ``region`` 為 ``(x, y, w, h)``，單位是**全片像素**；只有前兩個是原點，寬高由
    ``PrecutStream`` 在算格線時用掉，這裡不需要。
    """
    region = getattr(tile_stream, "region", None) if tile_stream is not None else None
    return (region[0], region[1]) if region else (0, 0)
