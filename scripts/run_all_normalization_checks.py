"""
Run all normalization verification checks at once
"""

import subprocess
import sys
from pathlib import Path
import argparse

def run_command(cmd, description):
    """Run a command and report results"""
    print("\n" + "="*70)
    print(f"Running: {description}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - FAILED")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ {description} - ERROR")
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run all normalization checks')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--config', type=str, required=True, help='Degradation config')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples for detailed check')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    print("="*70)
    print("NORMALIZATION VERIFICATION - COMPLETE TEST SUITE")
    print("="*70)
    print(f"\nData directory: {args.data_dir}")
    print(f"Config file: {args.config}")
    print(f"Number of samples: {args.num_samples}")
    
    results = []
    
    # Test 1: Quick verification (1 sample)
    cmd1 = [
        sys.executable,
        str(script_dir / 'verify_normalization.py'),
        '--data_dir', args.data_dir,
        '--config', args.config
    ]
    results.append(run_command(cmd1, "Quick Verification (1 sample)"))
    
    # Test 2: Detailed statistics
    cmd2 = [
        sys.executable,
        str(script_dir / 'check_normalization.py'),
        '--data_dir', args.data_dir,
        '--config', args.config,
        '--num_samples', str(args.num_samples)
    ]
    results.append(run_command(cmd2, f"Detailed Statistics ({args.num_samples} samples)"))
    
    # Test 3: Create flow diagram
    cmd3 = [
        sys.executable,
        str(script_dir / 'visualize_normalization_flow.py')
    ]
    results.append(run_command(cmd3, "Normalization Flow Diagram"))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    test_names = [
        "Quick Verification",
        "Detailed Statistics",
        "Flow Diagram"
    ]
    
    for name, result in zip(test_names, results):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:30s} {status}")
    
    print("\n" + "="*70)
    
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("\nNormalization is working correctly:")
        print("  • Input and target use consistent normalization")
        print("  • Both are in [0, 1] range")
        print("  • Same intensity scaling")
        print("\nYou can proceed with training.")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease review the output above and check:")
        print("  • Data directory is correct")
        print("  • Config file is valid")
        print("  • MriWizard is properly installed")
    
    print("="*70)
    
    # List generated files
    print("\nGenerated files:")
    parent_dir = script_dir.parent
    
    files_to_check = [
        parent_dir / 'normalization_verification.png',
        parent_dir / 'normalization_flow_diagram.png'
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (not found)")
    
    print()
    
    return 0 if all(results) else 1

if __name__ == '__main__':
    sys.exit(main())

