"""
Convert MATERIA V4 .basemateria to GGUF format.
Compatible with llama.cpp, Ollama, LM Studio.
"""
import os, sys, pickle
import numpy as np
import gguf

GGUF_MAGIC = 0x46554747  # 'GGUF'
GGUF_VERSION = 3


def convert_basemateria_to_gguf(input_path, output_path):
    print(f"📦 Loading .basemateria: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    state_dict = data['state_dict']
    config = data['config']
    stats = data.get('stats', {})
    
    print(f"   Tensors: {len(state_dict)}")
    print(f"   Config: {config}")

    # Determine dimensions from state_dict
    tok_emb = state_dict.get('tok_emb.weight', state_dict.get('emb.weight'))
    if tok_emb is None:
        # Find vocab_size and dim from state dict
        for k, v in state_dict.items():
            if 'weight' in k and len(v.shape) == 2:
                if v.shape[0] in [32000, 32768, 1024, 208]:  # vocab size
                    tok_emb = v
                    break
    
    vocab_size = config.get('vocab_size', tok_emb.shape[0] if tok_emb is not None else 32768)
    dim = config.get('dim', tok_emb.shape[1] if tok_emb is not None else 1536)
    n_layers = sum(1 for k in state_dict.keys() if 'layers.' in k and 'weight' in k)
    n_layers = config.get('n_layers', 24)
    n_heads = config.get('n_heads', 24)
    n_kv = config.get('n_kv', 6)
    latent_dim = config.get('latent_dim', dim)
    
    # GGUF writer
    print(f"🔧 Creating GGUF: {output_path}")
    writer = gguf.GGUFWriter(output_path, "materia-v4")
    
    # ─── Metadata ───
    writer.add_architecture()
    writer.add_context_length(2048)
    writer.add_embedding_length(dim)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(dim * 4)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv)
    writer.add_file_type(1)  # FP32
    
    # Custom MATERIA metadata
    writer.add_int32("materia.latent_dim", latent_dim)
    writer.add_float32("materia.jepa_weight", config.get('K', 2.781042))
    writer.add_string("materia.version", config.get('version', 'V4-enhanced'))
    
    writer.add_tokenizer_model("bpe" if data.get('tokenizer_type') == 'bpe' else 'char')
    
    # ─── Write ALL tensors ───
    # Standard naming for common layers (GGUF compatible)
    # + full original state dict preserved as "materia.*" tensors
    written = set()
    standard_count = 0
    special_count = 0
    
    for k, v in state_dict.items():
        arr = np.array(v, dtype=np.float32) if hasattr(v, 'dtype') and v.dtype == np.object_ else np.array(v)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        
        # Determine if this is a "standard" layer (can be mapped for inference) or MATERIA-specific
        is_materia_special = any(x in k.lower() for x in [
            'jepa', 'snn', 'ssm', 'hsaq', 'synapsis', 'torus', 'hexagonal',
            'spectral', 'emb_to_jepa', 't_to_jepa', 'snn_to_jepa', 'ssm_to_jepa',
            'jepa_enc', 'jepa_pred', 'mu', 'gate', 'up', 'down', 'spectral',
            'spike', 'lif', 'surrogate', 'moe_'
        ])
        
        if is_materia_special:
            gguf_name = f"materia.{k.replace('.', '_')}"
            special_count += 1
        else:
            # Standard mapping
            gguf_name = k
            standard_count += 1
        
        writer.add_tensor(gguf_name, arr)
        written.add(k)
    
    print(f"   Tensores escritos: {standard_count} estándar + {special_count} MATERIA-especiales")
    
    # Write tokenizer if available
    if 'tokenizer' in data:
        stoi = data['tokenizer']
        # Create vocab mapping
        itos = {v: k for k, v in stoi.items()} if isinstance(list(stoi.values())[0], int) else stoi
        tokens = [itos.get(i, f'<unk{i}>') for i in range(vocab_size)]
        writer.add_token_list(tokens)
    
    writer.add_int32("materia.total_tensors", len(written))
    writer.add_int32("materia.standard_tensors", standard_count)
    writer.add_int32("materia.special_tensors", special_count)
    
    # ─── Write GGUF file ───
    print(f"💾 Escribiendo GGUF...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    file_size = os.path.getsize(output_path)
    print(f"✅ GGUF creado: {output_path}")
    print(f"   Tamaño: {file_size/1024/1024/1024:.2f} GB")
    print(f"   Tensores: {len(written)} estándar + {len(special_tensors)} especiales MATERIA")
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Uso: python3 convert_to_gguf.py <input.basemateria> <output.gguf>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"❌ No existe: {input_path}")
        sys.exit(1)
    
    result = convert_basemateria_to_gguf(input_path, output_path)
    print(f"\n🎯 Archivo listo: {result}")
