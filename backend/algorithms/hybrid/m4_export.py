"""
M4 公開 API — 純函式庫：渲染成陣列 + 寫全域表格。不寫 slide 級影像檔。

Available:
  render_overlay_image          → m4_module.overlay
  draw_tile_seam_edges          → m4_module.overlay
  export_tile_csv               → m4_module.csv
  export_summary_statistics     → m4_module.csv
"""
try:
    from .m4_module.csv import (
        export_summary_statistics,
        export_tile_csv,
    )
    from .m4_module.overlay import (
        draw_tile_seam_edges,
        render_overlay_image,
    )
except ImportError:
    from m4_module.csv import (
        export_summary_statistics,
        export_tile_csv,
    )
    from m4_module.overlay import (
        draw_tile_seam_edges,
        render_overlay_image,
    )

__all__ = [
    "draw_tile_seam_edges",
    "export_summary_statistics",
    "export_tile_csv",
    "render_overlay_image",
]
