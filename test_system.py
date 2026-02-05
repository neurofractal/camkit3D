"""
Test script to verify system setup and camera availability
"""

import sys
import subprocess


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
        return False


def check_dependencies():
    """Check required Python packages"""
    print("\nChecking Python dependencies...")
    
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'scipy': 'scipy',
    }
    
    all_ok = True
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (run: pip install {package})")
            all_ok = False
    
    return all_ok


def check_ffmpeg():
    """Check if FFmpeg is installed"""
    print("\nChecking FFmpeg...")
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"  ✓ {version_line}")
            return True
        else:
            print("  ✗ FFmpeg not working properly")
            return False
            
    except FileNotFoundError:
        print("  ✗ FFmpeg not found")
        print("    Install from: https://ffmpeg.org/download.html")
        return False


def check_cameras():
    """Detect available cameras"""
    print("\nDetecting cameras...")
    
    try:
        import cv2
        
        cameras_found = []
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify it works
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"  ✓ Camera {i} ({width}x{height})")
                    cameras_found.append(i)
                cap.release()
        
        if cameras_found:
            print(f"\n  Total cameras found: {len(cameras_found)}")
            print(f"  Camera IDs: {cameras_found}")
            return True
        else:
            print("  ✗ No cameras found")
            print("    - Check camera connections")
            print("    - Close other applications using cameras")
            return False
            
    except Exception as e:
        print(f"  ✗ Error checking cameras: {e}")
        return False


def main():
    """Run all checks"""
    print("="*60)
    print("Multi-Camera Recorder - System Check")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("FFmpeg", check_ffmpeg),
        ("Cameras", check_cameras),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Error in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✓ All checks passed! System is ready.")
        print("\nNext steps:")
        print("  1. Run: python demo.py")
        print("  2. Choose a demo workflow")
        print("  3. Follow the instructions")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Install FFmpeg: https://ffmpeg.org/download.html")
        print("  - Check camera connections and close other camera apps")
    
    print("="*60)


if __name__ == "__main__":
    main()
