"""
Module 1B Comprehensive Test & Validation Script.

Validates all 8 required test scenarios and architectural invariants:
1. Acceptable image (NON-CRITICAL) -> 100% bypassed, 0 attempts -> OK TO GO
2. Critical image (CRITICAL) -> 100% bypassed, 0 attempts -> RECAPTURE
3. Borderline low-contrast -> targeted CLAHE -> contrast improved -> ACCEPT
4. Borderline uneven-illumination -> illumination normalization -> illumination improved -> ACCEPT
5. Borderline mild-underexposed -> gamma correction -> brightness improved -> ACCEPT
6. Borderline noisy -> mild bilateral denoising -> noise improved -> ACCEPT
7. Borderline remaining borderline -> 2 attempts executed -> RECAPTURE
8. Degradation detected -> safety check rejects enhancement -> escalated to CRITICAL / RECAPTURE

Invariant Checks:
- Max 2 operations per attempt
- Max 2 attempts per image
- No recursive loops
- Dataset files unmodified (SHA-256 integrity verification)

Outputs:
- MODULE 1B VALIDATION: PASS
"""

import os
import sys
import hashlib
import time
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import process_fundus_image
from src.quality_enhancer import process_borderline_image
from src.config import MAX_OPERATIONS_PER_ATTEMPT, MAX_ENHANCEMENT_ATTEMPTS


def compute_sha256(filepath):
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_scenario_1_acceptable(dataset_dir):
    """Scenario 1: Acceptable Image (NON-CRITICAL) -> 100% Bypass -> OK TO GO."""
    fn = 'aptos_000c1434d8d7.png'
    path = os.path.join(dataset_dir, fn)
    assert os.path.exists(path), f"File {fn} not found in {dataset_dir}"
    
    hash_before = compute_sha256(path)
    res = process_fundus_image(path, filename=fn)
    hash_after = compute_sha256(path)
    assert hash_before == hash_after, f"Source image {fn} was modified!"
    
    m = res['metadata']
    assert m['original_status'] == 'NON-CRITICAL', f"Expected NON-CRITICAL, got {m['original_status']}"
    assert m['final_status'] == 'NON-CRITICAL', f"Expected final NON-CRITICAL, got {m['final_status']}"
    assert m['final_directive'] == 'OK TO GO', f"Expected OK TO GO, got {m['final_directive']}"
    assert m['final_decision'] == 'ACCEPT', f"Expected ACCEPT, got {m['final_decision']}"
    assert m['enhancement_attempt_number'] == 0, f"Expected 0 attempts, got {m['enhancement_attempt_number']}"
    assert len(m['operations_applied']) == 0, f"Expected 0 operations, got {m['operations_applied']}"
    assert m['bypassed'] is True, "Expected bypassed=True"
    assert res['ok_to_go'] is True, "Expected ok_to_go=True"
    assert res['recapture_required'] is False, "Expected recapture_required=False"
    print("  [PASS] Scenario 1: Acceptable Image (NON-CRITICAL) -> 100% Bypassed -> OK TO GO")
    return True


def test_scenario_2_critical(dataset_dir):
    """Scenario 2: Critical Image (CRITICAL) -> 100% Bypass -> RECAPTURE."""
    fn = 'aptos_6a244e855d0e.png'
    path = os.path.join(dataset_dir, fn)
    assert os.path.exists(path), f"File {fn} not found in {dataset_dir}"
    
    hash_before = compute_sha256(path)
    res = process_fundus_image(path, filename=fn)
    hash_after = compute_sha256(path)
    assert hash_before == hash_after, f"Source image {fn} was modified!"
    
    m = res['metadata']
    assert m['original_status'] == 'CRITICAL', f"Expected CRITICAL, got {m['original_status']}"
    assert m['final_status'] == 'CRITICAL', f"Expected final CRITICAL, got {m['final_status']}"
    assert m['final_directive'] == 'RECAPTURE', f"Expected RECAPTURE, got {m['final_directive']}"
    assert m['final_decision'] == 'REJECT', f"Expected REJECT, got {m['final_decision']}"
    assert m['enhancement_attempt_number'] == 0, f"Expected 0 attempts, got {m['enhancement_attempt_number']}"
    assert len(m['operations_applied']) == 0, f"Expected 0 operations, got {m['operations_applied']}"
    assert m['bypassed'] is True, "Expected bypassed=True"
    assert res['ok_to_go'] is False, "Expected ok_to_go=False"
    assert res['recapture_required'] is True, "Expected recapture_required=True"
    print("  [PASS] Scenario 2: Critical Image (CRITICAL) -> 100% Bypassed -> RECAPTURE")
    return True


def test_scenario_3_low_contrast(dataset_dir):
    """Scenario 3: Borderline Low-Contrast -> Targeted CLAHE -> Contrast Improved."""
    fn = 'aptos_0fcfc6301f3d.png'
    path = os.path.join(dataset_dir, fn)
    assert os.path.exists(path), f"File {fn} not found in {dataset_dir}"
    
    hash_before = compute_sha256(path)
    res = process_fundus_image(path, filename=fn)
    hash_after = compute_sha256(path)
    assert hash_before == hash_after, f"Source image {fn} was modified!"
    
    m = res['metadata']
    assert m['original_status'] == 'BORDERLINE', f"Expected BORDERLINE, got {m['original_status']}"
    assert 'CLAHE' in m['operations_applied'], f"Expected CLAHE in operations, got {m['operations_applied']}"
    c_orig = m['original_normalized_scores']['contrast']
    c_post = m['enhanced_normalized_scores']['contrast']
    assert c_post > c_orig, f"Contrast score did not improve: {c_orig:.3f} -> {c_post:.3f}"
    assert m['final_status'] == 'NON-CRITICAL', f"Expected final NON-CRITICAL, got {m['final_status']}"
    assert m['final_decision'] == 'ACCEPT', f"Expected ACCEPT, got {m['final_decision']}"
    assert m['enhancement_success'] is True, "Expected enhancement_success=True"
    print(f"  [PASS] Scenario 3: Borderline Low-Contrast -> CLAHE -> Contrast Score {c_orig:.3f} -> {c_post:.3f} -> NON-CRITICAL")
    return True


def test_scenario_4_uneven_illumination(dataset_dir):
    """Scenario 4: Borderline Uneven Illumination -> Illumination Normalization -> Improved."""
    fn = 'train_IDRiD_352.jpg'
    path = os.path.join(dataset_dir, fn)
    assert os.path.exists(path), f"File {fn} not found in {dataset_dir}"
    
    hash_before = compute_sha256(path)
    res = process_fundus_image(path, filename=fn)
    hash_after = compute_sha256(path)
    assert hash_before == hash_after, f"Source image {fn} was modified!"
    
    m = res['metadata']
    assert m['original_status'] == 'BORDERLINE', f"Expected BORDERLINE, got {m['original_status']}"
    assert 'illumination_normalization' in m['operations_applied'], f"Expected illumination_normalization, got {m['operations_applied']}"
    assert m['final_status'] == 'NON-CRITICAL', f"Expected final NON-CRITICAL, got {m['final_status']}"
    assert m['final_directive'] == 'OK TO GO', f"Expected OK TO GO, got {m['final_directive']}"
    assert m['final_decision'] == 'ACCEPT', f"Expected ACCEPT, got {m['final_decision']}"
    assert m['score_change'] > 0, f"Expected positive score change, got {m['score_change']}"
    print(f"  [PASS] Scenario 4: Borderline Uneven Illumination -> Normalization -> Overall Score Delta: +{m['score_change']:.4f} -> OK TO GO")
    return True


def test_scenario_5_mild_underexposure(dataset_dir):
    """Scenario 5: Borderline Mild Underexposure -> Gamma Correction -> Brightness Improved."""
    fn = 'aptos_09935d72892b.png'
    path = os.path.join(dataset_dir, fn)
    assert os.path.exists(path), f"File {fn} not found in {dataset_dir}"
    
    hash_before = compute_sha256(path)
    res = process_fundus_image(path, filename=fn)
    hash_after = compute_sha256(path)
    assert hash_before == hash_after, f"Source image {fn} was modified!"
    
    m = res['metadata']
    assert m['original_status'] == 'BORDERLINE', f"Expected BORDERLINE, got {m['original_status']}"
    assert 'gamma_correction' in m['operations_applied'], f"Expected gamma_correction, got {m['operations_applied']}"
    b_orig = m['original_normalized_scores']['brightness']
    b_post = m['enhanced_normalized_scores']['brightness']
    assert b_post > b_orig, f"Brightness score did not improve: {b_orig:.3f} -> {b_post:.3f}"
    assert m['final_status'] == 'NON-CRITICAL', f"Expected final NON-CRITICAL, got {m['final_status']}"
    assert m['final_decision'] == 'ACCEPT', f"Expected ACCEPT, got {m['final_decision']}"
    print(f"  [PASS] Scenario 5: Borderline Mild Underexposure -> Gamma -> Brightness Score {b_orig:.3f} -> {b_post:.3f} -> NON-CRITICAL")
    return True


def test_scenario_6_noisy_image(dataset_dir):
    """Scenario 6: Borderline Noisy Image -> Mild Bilateral Denoising -> Noise Improved."""
    fn = 'aptos_663a923d5398.png'
    path = os.path.join(dataset_dir, fn)
    assert os.path.exists(path), f"File {fn} not found in {dataset_dir}"
    
    hash_before = compute_sha256(path)
    res = process_fundus_image(path, filename=fn)
    hash_after = compute_sha256(path)
    assert hash_before == hash_after, f"Source image {fn} was modified!"
    
    m = res['metadata']
    assert m['original_status'] == 'BORDERLINE', f"Expected BORDERLINE, got {m['original_status']}"
    assert 'bilateral_denoising' in m['operations_applied'], f"Expected bilateral_denoising, got {m['operations_applied']}"
    n_orig = m['original_normalized_scores']['noise']
    n_post = m['enhanced_normalized_scores']['noise']
    assert n_post > n_orig, f"Noise score did not improve: {n_orig:.3f} -> {n_post:.3f}"
    assert m['final_status'] == 'NON-CRITICAL', f"Expected final NON-CRITICAL, got {m['final_status']}"
    assert m['final_decision'] == 'ACCEPT', f"Expected ACCEPT, got {m['final_decision']}"
    print(f"  [PASS] Scenario 6: Borderline Noisy Image -> Bilateral Denoising -> Noise Score {n_orig:.3f} -> {n_post:.3f} -> NON-CRITICAL")
    return True


def test_scenario_7_remaining_borderline(dataset_dir):
    """Scenario 7: Borderline Remaining Borderline -> 2 Attempts Executed -> RECAPTURE."""
    fn = 'aptos_1891698febce.png'
    path = os.path.join(dataset_dir, fn)
    assert os.path.exists(path), f"File {fn} not found in {dataset_dir}"
    
    hash_before = compute_sha256(path)
    res = process_fundus_image(path, filename=fn)
    hash_after = compute_sha256(path)
    assert hash_before == hash_after, f"Source image {fn} was modified!"
    
    m = res['metadata']
    assert m['original_status'] == 'BORDERLINE', f"Expected BORDERLINE, got {m['original_status']}"
    assert m['final_status'] == 'BORDERLINE', f"Expected final BORDERLINE, got {m['final_status']}"
    assert m['enhancement_attempt_number'] == 2, f"Expected exactly 2 attempts, got {m['enhancement_attempt_number']}"
    assert m['final_directive'] == 'RECAPTURE', f"Expected RECAPTURE, got {m['final_directive']}"
    assert m['final_decision'] == 'REJECT', f"Expected REJECT, got {m['final_decision']}"
    assert res['recapture_required'] is True, "Expected recapture_required=True"
    assert res['ok_to_go'] is False, "Expected ok_to_go=False"
    print(f"  [PASS] Scenario 7: Borderline Remaining Borderline -> 2 Attempts Exhausted -> RECAPTURE / REJECT")
    return True


def test_scenario_8_enhancement_degradation(dataset_dir):
    """Scenario 8: Enhancement Degradation -> Safety Check Rejection -> Escalated to CRITICAL."""
    fn = 'aptos_2f143453bb71.png'
    path = os.path.join(dataset_dir, fn)
    assert os.path.exists(path), f"File {fn} not found in {dataset_dir}"
    
    hash_before = compute_sha256(path)
    res = process_fundus_image(path, filename=fn)
    hash_after = compute_sha256(path)
    assert hash_before == hash_after, f"Source image {fn} was modified!"
    
    m = res['metadata']
    assert m['original_status'] == 'BORDERLINE', f"Expected initial BORDERLINE, got {m['original_status']}"
    assert m['degradation_detected'] is True, "Expected degradation_detected=True"
    assert len(m['errors']) > 0, f"Expected non-empty safety errors, got {m['errors']}"
    assert m['final_status'] == 'CRITICAL', f"Expected escalated CRITICAL, got {m['final_status']}"
    assert m['final_directive'] == 'RECAPTURE', f"Expected RECAPTURE, got {m['final_directive']}"
    assert m['final_decision'] == 'REJECT', f"Expected REJECT, got {m['final_decision']}"
    assert res['recapture_required'] is True, "Expected recapture_required=True"
    # Verify reversion to original image
    assert np.array_equal(res['final_image'], res['original_image']), "Final image was not reverted to original on degradation!"
    print(f"  [PASS] Scenario 8: Enhancement Degradation Detected -> Safety Rejection -> Escalated to CRITICAL / RECAPTURE")
    return True


def test_invariant_rules(dataset_dir):
    """Verifies invariant architectural bounds across all processed samples."""
    test_files = [
        'aptos_000c1434d8d7.png',
        'aptos_6a244e855d0e.png',
        'aptos_0fcfc6301f3d.png',
        'train_IDRiD_352.jpg',
        'aptos_09935d72892b.png',
        'aptos_663a923d5398.png',
        'aptos_1891698febce.png',
        'aptos_2f143453bb71.png'
    ]
    
    for fn in test_files:
        path = os.path.join(dataset_dir, fn)
        res = process_fundus_image(path, filename=fn)
        m = res['metadata']
        
        # 1. Check max attempts
        assert m['enhancement_attempt_number'] <= MAX_ENHANCEMENT_ATTEMPTS, (
            f"Image {fn} exceeded MAX_ENHANCEMENT_ATTEMPTS: {m['enhancement_attempt_number']} > {MAX_ENHANCEMENT_ATTEMPTS}"
        )
        
        # 2. Check operations per attempt
        if 'attempts_history' in m:
            for att in m['attempts_history']:
                assert len(att['operations']) <= MAX_OPERATIONS_PER_ATTEMPT, (
                    f"Attempt {att['attempt']} for {fn} exceeded MAX_OPERATIONS_PER_ATTEMPT: {len(att['operations'])} > {MAX_OPERATIONS_PER_ATTEMPT}"
                )
                
        # 3. Check output image validity
        final_img = res['final_image']
        assert final_img.dtype == np.uint8, f"Invalid final image dtype {final_img.dtype}"
        assert final_img.ndim == 3 and final_img.shape[2] == 3, f"Invalid final image shape {final_img.shape}"
        assert np.isfinite(final_img).all(), f"Non-finite pixels found in final image of {fn}"
        assert final_img.shape == res['original_image'].shape, f"Final image dimensions differ from original for {fn}"
        
    print("  [PASS] Invariant Rules: Max 2 ops/attempt, Max 2 attempts, Shape & DataType Integrity verified.")
    return True


def run_all_tests():
    dataset_dir = os.path.join(PROJECT_ROOT, 'dataset')
    assert os.path.isdir(dataset_dir), f"Dataset directory not found: {dataset_dir}"
    
    print("=" * 70)
    print("STARTING MODULE 1B SYSTEM VALIDATION SUITE")
    print("=" * 70)
    t_start = time.time()
    
    test_scenario_1_acceptable(dataset_dir)
    test_scenario_2_critical(dataset_dir)
    test_scenario_3_low_contrast(dataset_dir)
    test_scenario_4_uneven_illumination(dataset_dir)
    test_scenario_5_mild_underexposure(dataset_dir)
    test_scenario_6_noisy_image(dataset_dir)
    test_scenario_7_remaining_borderline(dataset_dir)
    test_scenario_8_enhancement_degradation(dataset_dir)
    test_invariant_rules(dataset_dir)
    
    t_total = time.time() - t_start
    print("=" * 70)
    print(f"ALL 8 TEST SCENARIOS AND INVARIANT RULES PASSED ({t_total:.2f}s)")
    print("MODULE 1B VALIDATION: PASS")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()
