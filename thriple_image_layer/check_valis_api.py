"""檢查 VALIS API"""
import valis

print("VALIS 版本:", valis.__version__)
print("\nVALIS 可用屬性:")
print([attr for attr in dir(valis) if not attr.startswith('_')])
