"""測試 Module 2 Alignment"""
from pathlib import Path
from config import create_default_config

def test_import():
    """測試能否正常 import"""
    from valis import non_rigid_registrars
    print("✓ VALIS non_rigid_registrars imported successfully")
    
    # 檢查可用的類別
    print(f"✓ OpticalFlowWarper available: {hasattr(non_rigid_registrars, 'OpticalFlowWarper')}")
    print(f"✓ NonRigidTileRegistrar available: {hasattr(non_rigid_registrars, 'NonRigidTileRegistrar')}")
    
def test_module2_import():
    """測試 module2 能否正常 import"""
    try:
        import module2_alignment
        print("✓ module2_alignment imported successfully")
    except Exception as e:
        print(f"✗ Failed to import module2_alignment: {e}")
        raise

def test_config():
    """測試配置是否正確"""
    config = create_default_config()
    print(f"✓ Config created")
    print(f"  - Output dir: {config.output_dir}")
    print(f"  - Modalities: {[m.name for m in config.modalities]}")
    
if __name__ == "__main__":
    print("=== Testing Module 2 ===\n")
    test_import()
    print()
    test_module2_import()
    print()
    test_config()
    print("\n=== All tests passed ===")
