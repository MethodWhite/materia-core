#!/usr/bin/env python3
"""Kino Chile video processing pipeline: frames -> OCR -> V-JEPA2 embeddings."""

import os, sys, json, re, time, subprocess, shutil
from pathlib import Path
from collections import defaultdict

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/tmp/rar_venv/lib/python3.14/site-packages')

import cv2
import torch

VIDEO_DIR = Path('/home/methodwhite/MATERIA/data/kino_videos')
FRAMES_DIR = Path('/home/methodwhite/MATERIA/data/kino_frames')
HISTORICAL_PATH = Path('/home/methodwhite/MATERIA/data/kino/historical/kino_mega.json')
MODEL_PATH = Path('/home/methodwhite/Proyectos/models/jepa/vjepa2-vitl')
OUTPUT_PATH = Path('/home/methodwhite/MATERIA/data/kino/embeddings/kino_vjepa_embeddings.json')
SAMPLE_DIR = Path('/tmp/kino_samples')
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

HISTORICAL_OFFSET = 3238


def get_video_info(video_path):
    stem = video_path.stem
    m = re.search(r'sorteo_(\d+)_(\d{4}-\d{2}-\d{2})', stem)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def extract_frames(video_path, output_dir, fps=1.0):
    if output_dir.exists():
        existing = list(output_dir.glob('frame_*.jpg'))
        if existing:
            return sorted(existing)
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which('ffmpeg'):
        subprocess.run([
            'ffmpeg', '-i', str(video_path),
            '-vf', f'fps={fps}', '-q:v', '2',
            '-y', str(output_dir / 'frame_%05d.jpg')
        ], capture_output=True)
    else:
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = int(round(video_fps / fps))
        saved = 0
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                cv2.imwrite(str(output_dir / f'frame_{saved:05d}.jpg'), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                saved += 1
            idx += 1
        cap.release()
    return sorted(output_dir.glob('frame_*.jpg'))


def create_digit_templates():
    templates = {}
    for digit in range(10):
        tmpl = np.zeros((30, 20), dtype=np.uint8)
        cv2.putText(tmpl, str(digit), (2, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
        templates[digit] = tmpl
    return templates


def find_numbers_in_roi(roi_gray, digit_templates, min_conf=0.25):
    _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    results = []
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if area < 15 or area > 1500:
            continue
        if ch < 4 or cw < 3:
            continue
        if ch > 50 or cw > 40:
            continue
        if cw / max(ch, 1) > 4.0:
            continue
        roi = binary[y:y+ch, x:x+cw]
        if roi.size == 0:
            continue
        try:
            roi_resized = cv2.resize(roi, (20, 30))
        except cv2.error:
            continue
        best_digit = -1
        best_score = -1
        for digit, tmpl in digit_templates.items():
            result = cv2.matchTemplate(roi_resized, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = score
                best_digit = digit
        if best_score >= min_conf:
            results.append({
                'x': x, 'y': y, 'w': cw, 'h': ch,
                'digit': best_digit, 'confidence': best_score, 'area': area
            })
    return results


def group_digits_into_numbers(detections, max_x_gap=25, max_y_gap=10):
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: (d['y'], d['x']))
    groups = []
    current = [dets[0]]
    for det in dets[1:]:
        prev = current[-1]
        if abs(det['y'] - prev['y']) < max_y_gap and (det['x'] - (prev['x'] + prev['w'])) < max_x_gap:
            current.append(det)
        else:
            groups.append(current)
            current = [det]
    if current:
        groups.append(current)
    numbers = []
    for group in groups:
        group.sort(key=lambda d: d['x'])
        digits = [d['digit'] for d in group if d['digit'] >= 0]
        conf = float(np.mean([d['confidence'] for d in group if d['digit'] >= 0]))
        if not digits:
            continue
        try:
            num = int(''.join(str(d) for d in digits))
            if 1 <= num <= 25 and conf > 0.25:
                numbers.append({
                    'number': num, 'confidence': conf,
                    'x': int(np.mean([d['x'] for d in group])),
                    'y': int(np.mean([d['y'] for d in group])),
                })
        except ValueError:
            continue
    return numbers


def ocr_video(frame_dir, digit_templates):
    frames = sorted(Path(frame_dir).glob('frame_*.jpg'))
    if not frames:
        return None, 'No frames'

    all_numbers = []

    for fp in frames[-40:]:
        img = cv2.imread(str(fp))
        if img is None or img.mean() < 5:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for region_name, y_start, y_end in [
            ('bottom', int(h*0.55), h),
            ('top', 0, int(h*0.25)),
            ('center', int(h*0.25), int(h*0.55)),
        ]:
            roi = gray[y_start:y_end, :]
            dets = find_numbers_in_roi(roi, digit_templates)
            nums = group_digits_into_numbers(dets)
            for n in nums:
                n['frame'] = fp.name
                n['region'] = region_name
            all_numbers.extend(nums)

    if not all_numbers:
        return None, 'No numbers detected'

    scores = defaultdict(float)
    for n in all_numbers:
        scores[n['number']] += n['confidence'] * 2.0

    sorted_nums = sorted(scores.items(), key=lambda x: -x[1])
    top14 = sorted([n for n, s in sorted_nums[:14]])
    avg_conf = np.mean([n['confidence'] for n in all_numbers if n['number'] in top14])

    return top14, f'confidence={avg_conf:.2f}'


def get_historical_draw(sorteo):
    hist = json.loads(HISTORICAL_PATH.read_text())
    idx = HISTORICAL_OFFSET - sorteo
    if 0 <= idx < len(hist):
        return sorted(hist[idx])
    return None


def load_vjepa_model():
    try:
        from transformers import AutoModel, AutoConfig
        config = AutoConfig.from_pretrained(str(MODEL_PATH))
        model = AutoModel.from_pretrained(str(MODEL_PATH))
        model = model.float()
        model.eval()
        return model
    except Exception as e:
        print(f'  V-JEPA2 load error: {e}')
        return None


def extract_embedding(model, frame_path, target_size=256):
    if model is None:
        return None
    try:
        img = cv2.imread(str(frame_path))
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (target_size, target_size))
        img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(1)
        with torch.no_grad():
            out = model(pixel_values_videos=img_tensor)
        embedding = out.last_hidden_state[:, 0, :].cpu().numpy().flatten().tolist()
        return embedding
    except Exception as e:
        print(f'  Embedding error: {e}')
        return None


def main():
    print('=' * 70)
    print('KINO CHILE - V-JEPA2 VIDEO PROCESSING PIPELINE')
    print('=' * 70)

    videos = sorted(VIDEO_DIR.glob('kino_sorteo_*.mp4'))
    videos = [v for v in videos if '_test.mp4' not in v.name]
    print(f'\nFound {len(videos)} videos')

    digit_templates = create_digit_templates()
    vjepa_model = load_vjepa_model()
    hist_data = json.loads(HISTORICAL_PATH.read_text())

    total_frames = 0
    total_ocr_correct = 0
    total_ocr_possible = 0
    results_list = []

    for vpath in videos:
        sorteo, date = get_video_info(vpath)
        if sorteo is None:
            continue

        frame_dir = FRAMES_DIR / str(sorteo)
        print(f'\n[{sorteo}] {date}')

        frames = extract_frames(vpath, frame_dir)
        total_frames += len(frames)

        expected = get_historical_draw(sorteo)
        if expected:
            print(f'  Expected: {expected}')

        ocr_start = time.time()
        ocr_numbers, ocr_info = ocr_video(frame_dir, digit_templates)
        ocr_time = time.time() - ocr_start

        if expected and ocr_numbers:
            correct = sum(1 for n in ocr_numbers if n in expected)
            total_ocr_correct += correct
            total_ocr_possible += len(expected)
            print(f'  OCR ({ocr_time:.1f}s): {correct}/{len(expected)} correct ({correct/len(expected):.0%})')
            print(f'    OCR nums: {ocr_numbers}')
        elif expected:
            total_ocr_possible += len(expected)
            print(f'  OCR ({ocr_time:.1f}s): 0/14 correct')
        else:
            print(f'  OCR ({ocr_time:.1f}s): {ocr_numbers} {ocr_info}')

        # Save a sample result frame
        sample_saved = False
        for fp in reversed(frames):
            img = cv2.imread(str(fp))
            if img is not None and img.mean() > 5 and not sample_saved:
                cv2.imwrite(str(SAMPLE_DIR / f'{sorteo}_result.jpg'), img)
                sample_saved = True

        embedding = None
        if vjepa_model and frames:
            # Use a frame from the result section (last 30 non-black)
            embed_frame = None
            for fp in reversed(frames):
                img = cv2.imread(str(fp))
                if img is not None and img.mean() > 5:
                    embed_frame = fp
                    break
            if embed_frame:
                embedding = extract_embedding(vjepa_model, embed_frame)
                print(f'  Embedding: {len(embedding) if embedding else 0} dims')

        results_list.append({
            'sorteo': sorteo,
            'date': date,
            'numbers': expected if expected else ocr_numbers,
            'embedding': embedding,
            'video': str(vpath),
            'frame_count': len(frames),
        })

    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'  Videos processed: {len(videos)}')
    print(f'  Total frames extracted: {total_frames}')
    if total_ocr_possible > 0:
        print(f'  OCR accuracy: {total_ocr_correct}/{total_ocr_possible} ({total_ocr_correct/total_ocr_possible:.1%})')
    emb_count = sum(1 for r in results_list if r['embedding'] is not None)
    print(f'  Embeddings generated: {emb_count}')
    print(f'  Sample frames saved to: {SAMPLE_DIR}')

    output_data = [{
        'sorteo': r['sorteo'],
        'date': r['date'],
        'numbers': r['numbers'] if r['numbers'] else None,
        'embedding': r['embedding'],
        'video': r['video'],
    } for r in results_list]

    OUTPUT_PATH.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f'\n  Results saved to {OUTPUT_PATH}')
    print(f'  File size: {OUTPUT_PATH.stat().st_size/1024:.0f} KB')


if __name__ == '__main__':
    main()
