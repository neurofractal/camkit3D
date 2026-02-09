"""
Test and Validation Script

This script validates the pipeline setup and tests individual components.
Run this before processing your actual data to ensure everything works.

Author: Generated for FreeMoCap-style workflow
Date: 2026-02-06
"""

import sys
import numpy as np
from pathlib import Path
import traceback


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def test_imports():
    """Test that all required packages can be imported."""
    print_section("Testing Package Imports")
    
    packages = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'scipy': 'SciPy',
        'toml': 'TOML',
        'tqdm': 'TQDM',
        'matplotlib': 'Matplotlib'
    }
    
    results = {}
    for module_name, display_name in packages.items():
        try:
            __import__(module_name)
            print(f"✓ {display_name:20s} - OK")
            results[module_name] = True
        except ImportError as e:
            print(f"✗ {display_name:20s} - MISSING")
            print(f"  Error: {e}")
            results[module_name] = False
    
    all_ok = all(results.values())
    
    if all_ok:
        print("\n✓ All required packages are installed!")
    else:
        print("\n✗ Some packages are missing. Install with:")
        print("  pip install -r requirements.txt")
    
    return all_ok


def test_module_imports():
    """Test that our custom modules can be imported."""
    print_section("Testing Custom Module Imports")
    
    modules = [
        'pose_processor',
        'pose_3d_projector',
        'run_pipeline'
    ]
    
    results = {}
    for module_name in modules:
        try:
            module = __import__(module_name)
            print(f"✓ {module_name:30s} - OK")
            results[module_name] = True
        except Exception as e:
            print(f"✗ {module_name:30s} - FAILED")
            print(f"  Error: {e}")
            results[module_name] = False
    
    all_ok = all(results.values())
    
    if all_ok:
        print("\n✓ All custom modules loaded successfully!")
    else:
        print("\n✗ Some modules failed to load.")
    
    return all_ok


def test_mediapipe_pose():
    """Test MediaPipe Pose initialization."""
    print_section("Testing MediaPipe Pose")
    
    try:
        import mediapipe as mp
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✓ MediaPipe Pose initialized successfully")
        
        # Test with a dummy image
        import cv2
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = pose.process(dummy_image)
        
        print("✓ MediaPipe Pose processing works")
        
        pose.close()
        
        return True
        
    except Exception as e:
        print(f"✗ MediaPipe Pose test failed: {e}")
        traceback.print_exc()
        return False


def test_calibration_loading():
    """Test camera calibration loading."""
    print_section("Testing Camera Calibration Loading")
    
    try:
        from pose_3d_projector import CameraCalibration
        import toml
        
        # Create a dummy calibration
        dummy_calib = {
            'name': 'test_camera',
            'size': [1280, 720],
            'matrix': [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]],
            'distortions': [0.1, 0, 0, 0, 0],
            'rotation': [0, 0, 0],
            'translation': [0, 0, 1000]
        }
        
        cam = CameraCalibration(**dummy_calib)
        
        print(f"✓ Camera calibration loaded: {cam.name}")
        print(f"  Size: {cam.size}")
        print(f"  Projection matrix shape: {cam.projection_matrix.shape}")
        
        # Test projection
        test_point = np.array([[0, 0, 1000]])
        projected = cam.project_points(test_point)
        
        print(f"✓ Point projection works: {test_point[0]} -> {projected[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Calibration test failed: {e}")
        traceback.print_exc()
        return False


def test_triangulation():
    """Test DLT triangulation."""
    print_section("Testing DLT Triangulation")
    
    try:
        from pose_3d_projector import CameraCalibration
        import cv2
        
        # Create two dummy cameras
        cam1 = CameraCalibration(
            name='cam1',
            size=[1280, 720],
            matrix=[[1000, 0, 640], [0, 1000, 360], [0, 0, 1]],
            distortions=[0, 0, 0, 0, 0],
            rotation=[0, 0, 0],
            translation=[0, 0, 0]
        )
        
        cam2 = CameraCalibration(
            name='cam2',
            size=[1280, 720],
            matrix=[[1000, 0, 640], [0, 1000, 360], [0, 0, 1]],
            distortions=[0, 0, 0, 0, 0],
            rotation=[0, np.pi/4, 0],
            translation=[500, 0, 0]
        )
        
        # Create a test 3D point
        true_point_3d = np.array([0, 0, 1000])
        
        # Project to 2D in both cameras
        point_2d_cam1 = cam1.project_points(true_point_3d.reshape(1, 3))
        point_2d_cam2 = cam2.project_points(true_point_3d.reshape(1, 3))
        
        print(f"✓ Test point: {true_point_3d}")
        print(f"  Cam1 projection: {point_2d_cam1[0]}")
        print(f"  Cam2 projection: {point_2d_cam2[0]}")
        
        # Build DLT matrix
        points_2d = np.array([point_2d_cam1[0], point_2d_cam2[0]])
        cameras = [cam1, cam2]
        
        A = []
        for point, camera in zip(points_2d, cameras):
            x, y = point
            P = camera.projection_matrix
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        
        A = np.array(A)
        
        # Solve
        _, _, Vt = np.linalg.svd(A)
        point_3d_hom = Vt[-1, :]
        reconstructed_point = point_3d_hom[:3] / point_3d_hom[3]
        
        # Check error
        error = np.linalg.norm(true_point_3d - reconstructed_point)
        
        print(f"  Reconstructed: {reconstructed_point}")
        print(f"  Reconstruction error: {error:.6f} mm")
        
        if error < 1.0:
            print("✓ DLT triangulation works accurately!")
            return True
        else:
            print(f"✗ High reconstruction error: {error} mm")
            return False
        
    except Exception as e:
        print(f"✗ Triangulation test failed: {e}")
        traceback.print_exc()
        return False


def test_data_formats():
    """Test that data can be saved and loaded correctly."""
    print_section("Testing Data I/O")
    
    try:
        # Test numpy save/load
        test_data = np.random.rand(100, 33, 3)
        
        test_file = Path("test_data.npy")
        np.save(test_file, test_data)
        loaded_data = np.load(test_file)
        
        assert np.allclose(test_data, loaded_data), "Data mismatch!"
        
        test_file.unlink()  # Clean up
        
        print("✓ NumPy save/load works")
        
        # Test TOML save/load
        import toml
        
        test_toml = {
            'test': {
                'value': 123,
                'array': [1, 2, 3]
            }
        }
        
        test_toml_file = Path("test.toml")
        with open(test_toml_file, 'w') as f:
            toml.dump(test_toml, f)
        
        with open(test_toml_file, 'r') as f:
            loaded_toml = toml.load(f)
        
        assert loaded_toml == test_toml, "TOML data mismatch!"
        
        test_toml_file.unlink()  # Clean up
        
        print("✓ TOML save/load works")
        
        return True
        
    except Exception as e:
        print(f"✗ Data I/O test failed: {e}")
        traceback.print_exc()
        return False


def print_system_info():
    """Print system information."""
    print_section("System Information")
    
    print(f"Python version: {sys.version}")
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except:
        pass
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except:
        pass
    
    try:
        import mediapipe as mp
        print(f"MediaPipe version: {mp.__version__}")
    except:
        pass
    
    try:
        import platform
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.machine()}")
    except:
        pass


def run_all_tests():
    """Run all validation tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "PIPELINE VALIDATION TESTS" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    
    print_system_info()
    
    results = {
        'Package Imports': test_imports(),
        'Module Imports': test_module_imports(),
        'MediaPipe Pose': test_mediapipe_pose(),
        'Calibration Loading': test_calibration_loading(),
        'DLT Triangulation': test_triangulation(),
        'Data I/O': test_data_formats()
    }
    
    print_section("Test Summary")
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} : {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("SUCCESS! All tests passed. Your pipeline is ready to use.")
        print("\nNext steps:")
        print("  1. Ensure you have synchronized videos in 'synchronized_videos/' folder")
        print("  2. Have your camera_calibration.toml file ready")
        print("  3. Run: python run_pipeline.py")
    else:
        print("ATTENTION: Some tests failed. Please resolve the issues above.")
        print("\nCommon solutions:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check that all .py files are in the same directory")
        print("  - Verify Python version >= 3.8")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
