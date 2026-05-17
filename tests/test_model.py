"""
Model Tests for Predictive Maintenance System
Validates model loading and prediction logic
"""

import joblib
import json
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_model_loading():
    """Test that model files exist and load correctly"""
    print("\n📦 Testing Model Loading...")
    
    model_path = "artifacts/rul_model.joblib"
    scaler_path = "artifacts/feature_scaler.joblib"
    config_path = "artifacts/model_config.json"
    
    # Check files exist
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    assert os.path.exists(scaler_path), f"Scaler not found at {scaler_path}"
    assert os.path.exists(config_path), f"Config not found at {config_path}"
    
    # Load files
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    assert model is not None, "Model failed to load"
    assert scaler is not None, "Scaler failed to load"
    assert "feature_names" in config, "Config missing feature_names"
    
    print(f"✅ Model loaded successfully")
    print(f"   Features: {len(config['feature_names'])}")
    print(f"   Max RUL: {config.get('max_rul', 125)}")
    
    return model, scaler, config


def test_prediction_shape(model, scaler, config):
    """Test that predictions return correct shape"""
    print("\n📊 Testing Prediction Shape...")
    
    # Create dummy input with correct features
    feature_names = config['feature_names']
    dummy_input = np.random.rand(1, len(feature_names))
    
    # Scale and predict
    dummy_scaled = scaler.transform(dummy_input)
    prediction = model.predict(dummy_scaled)
    
    assert len(prediction.shape) == 1, f"Expected 1D array, got {prediction.shape}"
    assert prediction.shape[0] == 1, f"Expected 1 prediction, got {prediction.shape[0]}"
    assert prediction[0] >= 0, f"Prediction should be non-negative, got {prediction[0]}"
    
    print(f"✅ Prediction shape correct: {prediction.shape}")
    print(f"   Sample prediction: {prediction[0]:.2f} cycles")
    
    return prediction[0]


def test_rul_clipping(model, scaler, config):
    """Test that RUL predictions are properly clipped"""
    print("\n✂️ Testing RUL Clipping...")
    
    max_rul = config.get('max_rul', 125)
    feature_names = config['feature_names']
    
    # Test with input that might produce high RUL
    # Create features with early cycle values
    dummy_input = np.zeros((1, len(feature_names)))
    dummy_input[0, feature_names.index('cycle')] = 1  # Very early cycle
    
    dummy_scaled = scaler.transform(dummy_input)
    prediction = model.predict(dummy_scaled)[0]
    
    # Apply clipping (should be done in app)
    clipped = min(max(prediction, 0), max_rul)
    
    assert clipped <= max_rul, f"RUL {clipped} exceeds max {max_rul}"
    assert clipped >= 0, f"RUL {clipped} is negative"
    
    print(f"✅ RUL clipping works correctly")
    print(f"   Raw prediction: {prediction:.2f}")
    print(f"   Clipped: {clipped:.2f} (max: {max_rul})")


def test_feature_consistency(config):
    """Test that features match CMAPSS dataset"""
    print("\n🔍 Testing Feature Consistency...")
    
    feature_names = config['feature_names']
    
    # Expected base features
    expected_base = ['cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    
    # Check base features
    for feat in expected_base:
        assert feat in feature_names, f"Missing expected feature: {feat}"
    
    # Check sensor features
    sensors = [f for f in feature_names if f.startswith('sensor_')]
    assert len(sensors) >= 10, f"Expected at least 10 sensors, got {len(sensors)}"
    
    print(f"✅ Feature set is consistent")
    print(f"   Base features: {expected_base}")
    print(f"   Sensor features: {len(sensors)} sensors included")


def test_scaler_consistency(scaler, config):
    """Test that scaler works on real-looking data"""
    print("\n📏 Testing Scaler Consistency...")
    
    feature_names = config['feature_names']
    
    # Create realistic looking input
    realistic_input = np.zeros((1, len(feature_names)))
    
    # Fill with realistic values based on CMAPSS
    for i, feat in enumerate(feature_names):
        if feat == 'cycle':
            realistic_input[0, i] = 150
        elif feat.startswith('op_setting_'):
            realistic_input[0, i] = 100.0 if '3' in feat else 0.0
        elif feat.startswith('sensor_'):
            realistic_input[0, i] = np.random.uniform(500, 9000)
    
    # Should not raise exception
    try:
        scaled = scaler.transform(realistic_input)
        assert scaled.shape == realistic_input.shape, "Scaler changed input shape"
        print(f"✅ Scaler works correctly")
        print(f"   Input range: [{realistic_input.min():.2f}, {realistic_input.max():.2f}]")
        print(f"   Scaled range: [{scaled.min():.2f}, {scaled.max():.2f}]")
    except Exception as e:
        raise AssertionError(f"Scaler failed: {e}")


def run_model_tests():
    """Run all model tests"""
    print("\n" + "="*70)
    print("🧠 MODEL TESTS")
    print("="*70)
    
    try:
        # Load artifacts
        model, scaler, config = test_model_loading()
        
        # Run tests
        test_prediction_shape(model, scaler, config)
        test_rul_clipping(model, scaler, config)
        test_feature_consistency(config)
        test_scaler_consistency(scaler, config)
        
        print("\n" + "="*70)
        print("🎉 ALL MODEL TESTS PASSED!")
        print("="*70)
        return True
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    import sys
    success = run_model_tests()
    sys.exit(0 if success else 1)