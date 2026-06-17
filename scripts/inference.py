"""
MATERIA V3 - Inference
Multilingual text generation + Audio upscaling
"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
os.environ['OMP_NUM_THREADS'] = '4'

def get_model():
    from materia_v3_full import MateriaV3Full, Tokenizer
    tok = Tokenizer()
    model = MateriaV3Full(vocab_size=tok.vocab_size, dim=256, n_layers=4)
    ckpt = os.path.join(os.path.dirname(__file__), '..', 'models', 'materia-v3-full-trained.pth')
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    model.eval()
    return model, tok

def generate_text(prompt, max_new=50, temp=0.7, top_p=0.9):
    model, tok = get_model()
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    gen = model.generate(ids, max_new=max_new, temp=temp, top_p=top_p)
    return tok.decode(gen[0].tolist())

def upscale_audio(mp3_path, output_path):
    from audio_upscaler import AudioUpscaler, upscale_file
    model = AudioUpscaler()
    ckpt = os.path.join(os.path.dirname(__file__), '..', 'models', 'audio_upscaler.pth')
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    return upscale_file(model, mp3_path, output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['text', 'audio'], default='text')
    parser.add_argument('--prompt', default='Hello, world')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--max-new', type=int, default=50)
    args = parser.parse_args()

    if args.mode == 'text':
        result = generate_text(args.prompt, max_new=args.max_new)
        print(f"Prompt: {args.prompt}")
        print(f"Output: {result}")

    elif args.mode == 'audio':
        inp = args.input or '/home/methodwhite/MATERIA/data/audio/compressed/kino_sorteo_3170_2026-01-02_oldlJWIC5jg.mp3'
        out = args.output or inp.replace('.mp3', '_upscaled.wav')
        result = upscale_audio(inp, out)
        print(f"Upscaled: {result}")
