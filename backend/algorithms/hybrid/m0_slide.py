"""
M0 公開 API — 玻片層：預切串流、縫合幾何、逐塊執行、多行程、斷點續跑。

``m0_module/`` 內部檔案彼此直接 import，絕不透過本檔（否則會成環）；本檔只給
``hybrid_pipeline`` / ``backend/api`` / 測試等**套件外**呼叫端使用。

Available:
  PrecutStream                  → m0_module.m0_reader
  TileGeometry                  → m0_module.m0_stitch
  compute_tile_geometry         → m0_module.m0_stitch
  core_crop_bounds              → m0_module.m0_stitch
  _STITCH_SCRATCH               → m0_module.m0_stitch
  _stitch_overlay_slide         → m0_module.m0_stitch
  _frozen_gc_generation         → m0_module.m0_tile_runner
  _init_unet_inferencer         → m0_module.m0_tile_runner
  _init_cellpose_segmenter      → m0_module.m0_tile_runner
  _init_dish_cellpose_segmenter → m0_module.m0_tile_runner
  _process_precut_tile_gpu      → m0_module.m0_tile_runner
  _process_precut_tile_cpu      → m0_module.m0_tile_runner
  prefetch_tile_reads           → m0_module.m0_tile_runner
  _run_tiles_multiprocess       → m0_module.m0_multiprocess
  LAST_MP_WORKER_TIMINGS        → m0_module.m0_multiprocess
  _CKPT_DIRNAME                 → m0_module.m0_checkpoint
  _checkpoint_load              → m0_module.m0_checkpoint
  _checkpoint_init              → m0_module.m0_checkpoint
  _checkpoint_save              → m0_module.m0_checkpoint
  _skip_completed               → m0_module.m0_checkpoint
"""
try:
    from .m0_module.m0_reader import PrecutStream
    from .m0_module.m0_stitch import (
        _STITCH_SCRATCH,
        TileGeometry,
        compute_tile_geometry,
        core_crop_bounds,
        _stitch_overlay_slide,
    )
    from .m0_module.m0_tile_runner import (
        _frozen_gc_generation,
        _init_cellpose_segmenter,
        _init_dish_cellpose_segmenter,
        _init_unet_inferencer,
        _process_precut_tile_cpu,
        _process_precut_tile_gpu,
        prefetch_tile_reads,
    )
    from .m0_module.m0_multiprocess import (
        LAST_MP_WORKER_TIMINGS,
        _run_tiles_multiprocess,
    )
    from .m0_module.m0_checkpoint import (
        _CKPT_DIRNAME,
        _checkpoint_init,
        _checkpoint_load,
        _checkpoint_save,
        _skip_completed,
    )
except ImportError:
    from m0_module.m0_reader import PrecutStream
    from m0_module.m0_stitch import (
        _STITCH_SCRATCH,
        TileGeometry,
        compute_tile_geometry,
        core_crop_bounds,
        _stitch_overlay_slide,
    )
    from m0_module.m0_tile_runner import (
        _frozen_gc_generation,
        _init_cellpose_segmenter,
        _init_dish_cellpose_segmenter,
        _init_unet_inferencer,
        _process_precut_tile_cpu,
        _process_precut_tile_gpu,
        prefetch_tile_reads,
    )
    from m0_module.m0_multiprocess import (
        LAST_MP_WORKER_TIMINGS,
        _run_tiles_multiprocess,
    )
    from m0_module.m0_checkpoint import (
        _CKPT_DIRNAME,
        _checkpoint_init,
        _checkpoint_load,
        _checkpoint_save,
        _skip_completed,
    )

__all__ = [
    "PrecutStream",
    "TileGeometry",
    "compute_tile_geometry",
    "core_crop_bounds",
    "_STITCH_SCRATCH",
    "_stitch_overlay_slide",
    "_frozen_gc_generation",
    "_init_unet_inferencer",
    "_init_cellpose_segmenter",
    "_init_dish_cellpose_segmenter",
    "_process_precut_tile_gpu",
    "_process_precut_tile_cpu",
    "prefetch_tile_reads",
    "_run_tiles_multiprocess",
    "LAST_MP_WORKER_TIMINGS",
    "_CKPT_DIRNAME",
    "_checkpoint_load",
    "_checkpoint_init",
    "_checkpoint_save",
    "_skip_completed",
]
