"""
Evaluation Script for Module 1B on a Representative Borderline Cohort (50 images).

Computes:
- Total borderline images evaluated
- Borderline -> NON-CRITICAL conversion rate
- Borderline -> CRITICAL rate
- Images remaining borderline
- Average quality score improvement
- Per-metric improvement
- Average attempts
- Enhancement Success Rate
- Average processing time
- Visual validation comparison panels into reports/module1b_visual_validation/
- Summary report in reports/module1b_evaluation.md
- Full cohort CSV in reports/module1b_evaluation.csv
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import process_fundus_image


def create_side_by_side_panel(orig_bgr, final_bgr, meta, output_path):
    """
    Creates an informative before/after comparison panel with metadata annotations.
    """
    h_orig, w_orig = orig_bgr.shape[:2]
    h_final, w_final = final_bgr.shape[:2]
    
    # Standardize display resolution for visual report (height ~ 600px)
    disp_h = 600
    scale_o = disp_h / h_orig
    scale_f = disp_h / h_final
    disp_orig = cv2.resize(orig_bgr, (int(w_orig * scale_o), disp_h), interpolation=cv2.INTER_AREA)
    disp_final = cv2.resize(final_bgr, (int(w_final * scale_f), disp_h), interpolation=cv2.INTER_AREA)
    
    # Combined canvas: banner top (110px), images side by side, footer (60px)
    banner_h = 110
    footer_h = 60
    total_w = disp_orig.shape[1] + disp_final.shape[1] + 10
    total_h = disp_h + banner_h + footer_h
    
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    canvas[:] = (26, 26, 26)  # Dark gray background
    
    # Place images
    canvas[banner_h:banner_h + disp_h, :disp_orig.shape[1]] = disp_orig
    canvas[banner_h:banner_h + disp_h, disp_orig.shape[1] + 10:] = disp_final
    
    # Header Banner
    fn = meta['filename']
    orig_st = meta['original_status']
    orig_sc = meta['original_quality_metrics'].get('overall_score', 0.0)
    # Check if overall_score is in normalized scores or classifier
    orig_sc = meta.get('original_overall_score', orig_sc)
    
    final_st = meta['final_status']
    final_sc = meta.get('post_enhancement_overall_score', 0.0)
    delta = meta['score_change']
    decision = meta['final_decision']
    directive = meta['final_directive']
    ops = ', '.join(meta['operations_applied']) if meta['operations_applied'] else 'None (Bypassed / Reverted)'
    
    # Text rendering helper
    def draw_text(img, text, pos, font_scale=0.6, color=(255, 255, 255), thickness=1):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        
    draw_text(canvas, f"MODULE 1B VALIDATION: {fn}", (20, 32), 0.75, (255, 255, 255), 2)
    draw_text(canvas, f"Decision: {decision} ({directive}) | Attempts: {meta['enhancement_attempt_number']} | Time: {meta['processing_time']*1000:.1f}ms", (20, 62), 0.60, (0, 255, 200) if decision == 'ACCEPT' else (100, 100, 255), 1)
    draw_text(canvas, f"Operations Applied: {ops}", (20, 92), 0.55, (200, 200, 200), 1)
    
    # Subtitles over images
    orig_label = f"ORIGINAL: {orig_st} (Score: {orig_sc:.3f})"
    final_label = f"ENHANCED: {final_st} (Score: {final_sc:.3f}, Delta: {delta:+.3f})"
    draw_text(canvas, orig_label, (20, banner_h + 30), 0.65, (0, 200, 255), 2)
    draw_text(canvas, final_label, (disp_orig.shape[1] + 30, banner_h + 30), 0.65, (0, 255, 0) if final_st == 'NON-CRITICAL' else (0, 140, 255), 2)
    
    # Footer
    draw_text(canvas, f"Rationale: {meta['rationale'][:120]}...", (20, total_h - 25), 0.48, (180, 180, 180), 1)
    
    cv2.imwrite(output_path, canvas)


def main():
    dataset_dir = os.path.join(PROJECT_ROOT, 'dataset')
    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    visual_dir = os.path.join(reports_dir, 'module1b_visual_validation')
    os.makedirs(visual_dir, exist_ok=True)
    
    # Load 4178 dataset summary to extract borderline cohort
    df_all = pd.read_csv(os.path.join(reports_dir, 'module1_full_results_4178.csv'))
    b_df = df_all[df_all['original_status'] == 'BORDERLINE'].copy()
    
    # Select 50 representative borderline images across various conditions
    # 1. CLAHE contrast candidates
    clahe_cands = b_df[b_df['enhancement_operations'].str.contains('clahe', case=False, na=False)]['filename'].tolist()
    # 2. Uneven illumination candidates
    illum_cands = b_df[b_df['enhancement_operations'].str.contains('illumination', case=False, na=False)]['filename'].tolist()
    # 3. Bilateral denoising candidates
    noise_cands = b_df[b_df['enhancement_operations'].str.contains('bilateral', case=False, na=False)]['filename'].tolist()
    # 4. Gamma underexposure candidates
    gamma_cands = b_df[b_df['enhancement_operations'].str.contains('gamma', case=False, na=False)]['filename'].tolist()
    # 5. Glare candidates
    glare_cands = b_df[b_df['enhancement_operations'].str.contains('glare', case=False, na=False)]['filename'].tolist()
    # 6. Remaining borderline / rejected candidates
    unrec_cands = b_df[b_df['final_status'] == 'BORDERLINE']['filename'].tolist()
    # 7. Escalated to critical candidates
    crit_cands = b_df[b_df['final_status'] == 'CRITICAL']['filename'].tolist()
    
    selected_fns = []
    def add_from_list(lst, count):
        added = 0
        for f in lst:
            if f not in selected_fns and os.path.exists(os.path.join(dataset_dir, f)):
                selected_fns.append(f)
                added += 1
                if added >= count:
                    break
                    
    add_from_list(clahe_cands, 8)
    add_from_list(illum_cands, 8)
    add_from_list(noise_cands, 8)
    add_from_list(gamma_cands, 10)
    add_from_list(glare_cands, 6)
    add_from_list(unrec_cands, 6)
    add_from_list(crit_cands, 4)
    
    # Fill up to 50 if needed
    for f in b_df['filename'].tolist():
        if len(selected_fns) >= 50:
            break
        if f not in selected_fns and os.path.exists(os.path.join(dataset_dir, f)):
            selected_fns.append(f)
            
    print(f"Selected {len(selected_fns)} representative borderline fundus images for cohort evaluation.")
    
    # Run evaluation
    records = []
    t_start_all = time.time()
    visual_sample_count = 0
    
    for idx, fn in enumerate(selected_fns, 1):
        p = os.path.join(dataset_dir, fn)
        t0 = time.time()
        res = process_fundus_image(p, filename=fn)
        elapsed = time.time() - t0
        
        m = res['metadata']
        orig_sc = m['original_normalized_scores']
        post_sc = m['enhanced_normalized_scores']
        
        row = {
            'filename': fn,
            'original_status': m['original_status'],
            'original_score': res['original_overall_score'],
            'final_status': m['final_status'],
            'final_score': res['post_enhancement_overall_score'],
            'score_delta': m['score_change'],
            'final_decision': m['final_decision'],
            'final_directive': m['final_directive'],
            'enhancement_success': m['enhancement_success'],
            'degradation_detected': m['degradation_detected'],
            'attempts': m['enhancement_attempt_number'],
            'operations': '; '.join(m['operations_applied']) if m['operations_applied'] else 'None',
            'orig_focus': orig_sc['focus'],
            'orig_brightness': orig_sc['brightness'],
            'orig_contrast': orig_sc['contrast'],
            'orig_noise': orig_sc['noise'],
            'orig_illumination': orig_sc['illumination'],
            'orig_artifact': orig_sc['artifact'],
            'post_focus': post_sc['focus'],
            'post_brightness': post_sc['brightness'],
            'post_contrast': post_sc['contrast'],
            'post_noise': post_sc['noise'],
            'post_illumination': post_sc['illumination'],
            'post_artifact': post_sc['artifact'],
            'delta_focus': round(post_sc['focus'] - orig_sc['focus'], 4),
            'delta_brightness': round(post_sc['brightness'] - orig_sc['brightness'], 4),
            'delta_contrast': round(post_sc['contrast'] - orig_sc['contrast'], 4),
            'delta_noise': round(post_sc['noise'] - orig_sc['noise'], 4),
            'delta_illumination': round(post_sc['illumination'] - orig_sc['illumination'], 4),
            'delta_artifact': round(post_sc['artifact'] - orig_sc['artifact'], 4),
            'processing_time_s': round(elapsed, 4)
        }
        records.append(row)
        
        # Save side-by-side comparison for up to 12 diverse representative cases
        if visual_sample_count < 12:
            panel_fn = f"panel_{idx:02d}_{fn.replace('.png', '').replace('.jpg', '')}.jpg"
            create_side_by_side_panel(
                res['original_image'],
                res['final_image'],
                res,
                os.path.join(visual_dir, panel_fn)
            )
            visual_sample_count += 1
            
        print(f"[{idx:02d}/50] {fn}: {row['original_score']:.3f} -> {row['final_score']:.3f} ({row['score_delta']:+.3f}) | {row['final_status']} ({row['final_decision']}) | Ops: {row['operations']} | {elapsed:.2f}s")
        
    df_eval = pd.DataFrame(records)
    csv_out = os.path.join(reports_dir, 'module1b_evaluation.csv')
    df_eval.to_csv(csv_out, index=False)
    print(f"\nSaved cohort evaluation data to: {csv_out}")
    
    # Statistical Aggregation
    total_imgs = len(df_eval)
    conv_noncrit = len(df_eval[df_eval['final_status'] == 'NON-CRITICAL'])
    conv_crit = len(df_eval[df_eval['final_status'] == 'CRITICAL'])
    rem_borderline = len(df_eval[df_eval['final_status'] == 'BORDERLINE'])
    success_rate = (conv_noncrit / total_imgs) * 100.0
    crit_rate = (conv_crit / total_imgs) * 100.0
    rem_rate = (rem_borderline / total_imgs) * 100.0
    
    avg_score_orig = df_eval['original_score'].mean()
    avg_score_final = df_eval['final_score'].mean()
    avg_delta_all = df_eval['score_delta'].mean()
    avg_delta_conv = df_eval[df_eval['final_status'] == 'NON-CRITICAL']['score_delta'].mean()
    
    avg_attempts = df_eval['attempts'].mean()
    avg_proc_time = df_eval['processing_time_s'].mean()
    
    # Per metric delta
    mean_deltas = {
        'Focus': df_eval['delta_focus'].mean(),
        'Brightness': df_eval['delta_brightness'].mean(),
        'Contrast': df_eval['delta_contrast'].mean(),
        'Noise': df_eval['delta_noise'].mean(),
        'Illumination': df_eval['delta_illumination'].mean(),
        'Artifact': df_eval['delta_artifact'].mean()
    }
    
    # Generate Markdown Report
    md_out = os.path.join(reports_dir, 'module1b_evaluation.md')
    with open(md_out, 'w') as f:
        f.write("# Module 1B: Borderline Quality Enhancement Evaluation Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"- **Evaluated Cohort Size**: {total_imgs} representative borderline fundus images\n")
        f.write(f"- **Borderline -> NON-CRITICAL Conversions**: {conv_noncrit} ({success_rate:.1f}%)\n")
        f.write(f"- **Borderline Remaining BORDERLINE**: {rem_borderline} ({rem_rate:.1f}%)\n")
        f.write(f"- **Borderline -> CRITICAL (Escalated/Degradation)**: {conv_crit} ({crit_rate:.1f}%)\n")
        f.write(f"- **Enhancement Success Rate**: **{success_rate:.1f}%**\n")
        f.write(f"- **Average Quality Score**: {avg_score_orig:.4f} -> {avg_score_final:.4f} (Net: {avg_delta_all:+.4f})\n")
        f.write(f"- **Average Improvement on Converted Images**: **+{avg_delta_conv:+.4f}**\n")
        f.write(f"- **Average Enhancement Attempts**: {avg_attempts:.2f} attempts / image\n")
        f.write(f"- **Average Processing Time**: {avg_proc_time:.3f} s / image ({avg_proc_time*1000:.1f} ms)\n\n")
        
        f.write("## Per-Dimension Impact Analysis\n\n")
        f.write("| Quality Dimension | Average Delta Across Cohort | Status Impact |\n")
        f.write("| :--- | :---: | :--- |\n")
        for dim, val in mean_deltas.items():
            f.write(f"| **{dim}** | {val:+.4f} | {'Significant Gain' if val > 0.05 else ('Moderate Gain' if val > 0 else 'Neutral/Preserved')} |\n")
            
        f.write("\n## Conversion Breakdown by Primary Deficit\n\n")
        f.write("| Category | Evaluated Images | Converted to NON-CRITICAL | Success Rate |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        categories = [
            ('Contrast Deficit (CLAHE)', df_eval[df_eval['operations'].str.contains('CLAHE')]),
            ('Illumination Deficit (Flat-Fielding)', df_eval[df_eval['operations'].str.contains('illumination')]),
            ('Exposure Deficit (Gamma Correction)', df_eval[df_eval['operations'].str.contains('gamma')]),
            ('Sensor Noise Deficit (Bilateral)', df_eval[df_eval['operations'].str.contains('bilateral')]),
            ('Specular Glare (Inpainting)', df_eval[df_eval['operations'].str.contains('glare')])
        ]
        for cat_name, sub_df in categories:
            cnt = len(sub_df)
            conv = len(sub_df[sub_df['final_status'] == 'NON-CRITICAL'])
            pct = (conv / max(1, cnt)) * 100.0
            f.write(f"| {cat_name} | {cnt} | {conv} | {pct:.1f}% |\n")
            
        f.write("\n## Representative Visual Validation Panels\n\n")
        f.write(f"Generated {visual_sample_count} side-by-side comparison panels saved to `reports/module1b_visual_validation/`.\n\n")
        f.write("Each visual panel displays:\n")
        f.write("1. Left: Original image with initial score and identified borderline deficit\n")
        f.write("2. Right: Final image with post-enhancement score, score delta, operations applied, and decision (ACCEPT/REJECT)\n")
        f.write("3. Top/Bottom Banner: Processing duration, attempt number, safety audit result, and clinical rationale\n")
        
    print(f"\nSaved comprehensive evaluation report to: {md_out}")
    print("=" * 60)
    print(f"COHORT EVALUATION COMPLETED:")
    print(f"  Success Rate: {success_rate:.1f}% ({conv_noncrit}/{total_imgs})")
    print(f"  Avg Score Delta: {avg_delta_all:+.4f} (Converted: +{avg_delta_conv:+.4f})")
    print(f"  Avg Time: {avg_proc_time:.3f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
