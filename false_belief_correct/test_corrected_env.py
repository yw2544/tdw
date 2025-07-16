#!/usr/bin/env python3
"""
test environment
"""

from room_env import create_room_environment_correct

def main():
    """test environment"""
    print("🧪 test environment")
    print("=" * 60)
    
    try:
        # 创建环境
        result = create_room_environment_correct(
            num_colored_grids=5,
            output_dir="./test_output"
        )
        
        if result:
            print("\n✅ test success!")
            print(f"target object: {result['target_object']}")
            print(f"agent position: {result['agent_position']}")
            print(f"colored grids: {len(result['colored_positions'])}")
            print(f"view metrics: {len(result['view_metrics'])} 个")
        else:
            print("\n❌ test failed")
            
    except Exception as e:
        print(f"\n❌ test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 