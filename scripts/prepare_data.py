"""
MATERIA V3 - Data Pipeline for Real Datasets
Descarga y prepara datasets reales via HuggingFace.

Uso:
  python scripts/prepare_data.py --download c4 --size 100M
  python scripts/prepare_data.py --train-tokenizer
  python scripts/prepare_data.py --list
"""
import os, sys, json, gzip, argparse, time
from pathlib import Path

MATERIA_HOME = os.environ.get(
    'MATERIA_HOME',
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
)
DATA_DIR = os.path.join(MATERIA_HOME, 'data', 'multilingual', 'tokenizer')
os.makedirs(DATA_DIR, exist_ok=True)

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_datasets():
    """Muestra datasets disponibles y su estado."""
    datasets = {
        'c4': {
            'path': 'allenai/c4',
            'config': 'en',
            'splits': ['train', 'validation'],
            'size': '~800GB',
            'desc': 'Colossal Clean Crawled Corpus (EN)',
            'local': os.path.exists(os.path.join(DATA_DIR, 'c4_en.txt')),
        },
        'c4_sample': {
            'path': 'allenai/c4',
            'config': 'en',
            'splits': ['train'],
            'size': '~100MB',
            'desc': 'C4 sample (primeros 100K documentos)',
            'local': os.path.exists(os.path.join(DATA_DIR, 'c4_en_sample.json.gz')),
        },
        'wikipedia': {
            'path': 'wikipedia',
            'config': '20220301.en',
            'splits': ['train'],
            'size': '~20GB',
            'desc': 'Wikipedia EN completo',
            'local': False,
        },
        'wikipedia_multi': {
            'path': 'wikipedia',
            'config': None,
            'splits': ['train'],
            'size': '~50GB (12 idiomas)',
            'desc': 'Wikipedia multi-lenguaje',
            'local': any(f.startswith('wiki_') for f in os.listdir(DATA_DIR))
                    if os.path.exists(DATA_DIR) else False,
        },
        'the_stack': {
            'path': 'bigcode/the-stack',
            'config': 'default',
            'splits': ['train'],
            'size': '~3TB',
            'desc': 'Codigo fuente (GH)',
            'local': False,
        },
        'fineweb': {
            'path': 'HuggingFaceFW/fineweb',
            'config': 'CC-MAIN-2024-10',
            'splits': ['train'],
            'size': '~10TB',
            'desc': 'Dataset curado de alta calidad (edu)',
            'local': False,
        },
    }
    return datasets


def estimate_lines(size_str):
    mapping = {'100M': 100_000, '500M': 500_000, '1B': 1_000_000, '10B': 10_000_000}
    return mapping.get(size_str.upper(), 100_000)


def download_c4(sample=False, max_lines=100_000):
    """Descarga C4 (o una muestra) y lo guarda como texto plano."""
    try:
        from datasets import load_dataset
    except ImportError:
        log("ERROR: pip install datasets")
        sys.exit(1)

    log(f"Downloading C4 {'sample' if sample else 'dataset'}...")
    split = 'train' if sample else 'train'
    if sample:
        dataset = load_dataset('allenai/c4', 'en', split=split, streaming=True)
        count = 0
        with open(os.path.join(DATA_DIR, 'c4_en.txt'), 'w', encoding='utf-8') as f:
            for i, example in enumerate(dataset):
                text = example['text'].strip()
                if len(text) > 100:
                    f.write(text + '\n')
                    count += 1
                    if count >= max_lines:
                        break
                if (i + 1) % 10000 == 0:
                    log(f"  {i+1} documents processed, {count} saved")
        log(f"Saved {count} documents to c4_en.txt")
    else:
        dataset = load_dataset('allenai/c4', 'en', split=split, streaming=True)
        out_path = os.path.join(DATA_DIR, 'c4_en.txt')
        count = 0
        with open(out_path, 'w', encoding='utf-8') as f:
            for i, example in enumerate(dataset):
                text = example['text'].replace('\n', ' ').strip()
                if len(text) > 50:
                    f.write(text + '\n')
                    count += 1
                    if count >= max_lines:
                        break
                if (i + 1) % 50000 == 0:
                    log(f"  {i+1} docs, {count} saved ({os.path.getsize(out_path)//1024**2}MB)")
        log(f"Saved {count} docs ({os.path.getsize(out_path)//1024**2}MB)")


def download_wikipedia(languages=None, max_lines=500_000):
    try:
        from datasets import load_dataset
    except ImportError:
        log("ERROR: pip install datasets"); sys.exit(1)

    languages = languages or ['en', 'es', 'fr', 'de', 'pt', 'ar']
    log(f"Downloading Wikipedia: {languages}")
    for lang in languages:
        try:
            dataset = load_dataset('wikipedia', f'20220301.{lang}',
                                    split='train', streaming=True)
            out_path = os.path.join(DATA_DIR, f'wiki_{lang}.txt')
            count = 0
            with open(out_path, 'w', encoding='utf-8') as f:
                for example in dataset:
                    text = example['text'].replace('\n', ' ').strip()
                    if len(text) > 50:
                        f.write(text + '\n')
                        count += 1
                        if count >= max_lines:
                            break
            log(f"  {lang}: {count} articles -> {out_path}")
        except Exception as e:
            log(f"  {lang}: error - {e}")


def train_bpe_tokenizer(vocab_size=32768):
    """Entrena tokenizer BPE desde los datos descargados."""
    try:
        import sentencepiece as spm
    except ImportError:
        log("ERROR: pip install sentencepiece"); sys.exit(1)

    input_files = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.txt') and not fname.startswith('combined'):
            input_files.append(os.path.join(DATA_DIR, fname))

    # Create combined file for training
    combined_path = os.path.join(DATA_DIR, 'combined_for_spm.txt')
    if not os.path.exists(combined_path) or os.path.getsize(combined_path) < 1_000_000:
        log("Creating combined file for SPM training...")
        with open(combined_path, 'w', encoding='utf-8') as out:
            total = 0
            for fp in input_files:
                with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        out.write(line)
                        total += 1
                        if total >= 5_000_000:
                            break
        log(f"Combined: {os.path.getsize(combined_path)//1024**2}MB, {total} lines")

    model_prefix = f'materia_multilingual_v3'
    log(f"Training BPE tokenizer (vocab_size={vocab_size})...")

    spm.SentencePieceTrainer.Train(
        input=combined_path,
        model_prefix=os.path.join(DATA_DIR, model_prefix),
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type='bpe',
        max_sentence_length=8192,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        train_extremely_large_corpus=True,
        split_by_unicode_script=True,
        byte_fallback=True,
        num_threads=max(1, os.cpu_count() - 1),
    )
    log(f"Tokenizer trained: {DATA_DIR}/{model_prefix}.model")
    log(f"  Vocab size: {vocab_size}")
    log(f"  Type: BPE")


def download_fineweb(max_lines=1_000_000):
    """FineWeb: dataset curado de alta calidad educativa."""
    try:
        from datasets import load_dataset
    except ImportError:
        log("ERROR: pip install datasets"); sys.exit(1)

    log("Downloading FineWeb (CC-MAIN-2024-10)...")
    dataset = load_dataset('HuggingFaceFW/fineweb',
                            'CC-MAIN-2024-10', split='train', streaming=True)
    out_path = os.path.join(DATA_DIR, 'fineweb_en.txt')
    count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            text = example['text'].strip()
            if len(text) > 100:
                f.write(text + '\n')
                count += 1
                if count >= max_lines:
                    break
            if (i + 1) % 50000 == 0:
                log(f"  {i+1} docs, {count} saved ({os.path.getsize(out_path)//1024**2}MB)")
    log(f"FineWeb: {count} docs -> {out_path}")


def analyze_data():
    """Analiza los datos descargados: conteo de lineas, tokens, etc."""
    log("Data analysis:")
    total_lines = 0
    for fname in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, fname)
        if fname.endswith('.txt') and os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / 1024**2
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = sum(1 for _ in f)
            total_lines += lines
            log(f"  {fname}: {lines:>10,} lines, {size_mb:>7.1f}MB")
    log(f"  TOTAL: {total_lines:,} lines")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', type=str, default=None,
                        choices=['c4', 'c4_sample', 'wikipedia', 'fineweb', 'all'])
    parser.add_argument('--size', type=str, default='100M',
                        help='Tamaño: 100M, 500M, 1B (lineas)')
    parser.add_argument('--train-tokenizer', action='store_true')
    parser.add_argument('--vocab-size', type=int, default=32768)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()

    if args.list:
        datasets = list_datasets()
        print(f"\n{'Dataset':<20} {'Local':<8} {'Size':<15} Description")
        print('-' * 70)
        for name, info in datasets.items():
            local = '✓' if info['local'] else '✗'
            print(f"{name:<20} {local:<8} {info['size']:<15} {info['desc']}")
        return

    if args.analyze:
        analyze_data()
        return

    if args.download:
        max_lines = estimate_lines(args.size)
        if args.download in ('c4', 'all'):
            download_c4(sample=True, max_lines=max_lines)
        if args.download == 'c4_sample':
            download_c4(sample=True, max_lines=100_000)
        if args.download in ('wikipedia', 'all'):
            download_wikipedia(max_lines=min(max_lines, 500_000))
        if args.download in ('fineweb', 'all'):
            download_fineweb(max_lines=max_lines)
        if args.download == 'all':
            train_bpe_tokenizer(args.vocab_size)
        analyze_data()

    if args.train_tokenizer:
        train_bpe_tokenizer(args.vocab_size)

    if not args.download and not args.train_tokenizer and not args.list and not args.analyze:
        parser.print_help()


if __name__ == '__main__':
    main()
