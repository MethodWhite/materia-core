"""
Convert MATERIA V4 .basemateria to standard Llama-compatible GGUF.
Only extracts transformer layers (GQA) - drops JEPA/SNN/SSM/HSAQ.
Compatible with LM Studio, Ollama, llama.cpp.
"""
import os, sys, pickle
import numpy as np
import gguf

def convert_to_compatible_gguf(input_path, output_path):
    print(f"📦 Loading .basemateria: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    state_dict = data['state_dict']
    config = data['config']
    
    vocab_size = config.get('vocab_size', 32768)
    dim = config.get('dim', 1792)
    n_layers = sum(1 for k in state_dict if k.startswith('layers.') and k.endswith('.attention_norm.weight'))
    n_heads = config.get('n_heads', 24)
    n_kv = config.get('n_kv', 6)
    head_dim = dim // n_heads
    
    print(f"   Model: {config['version']}")
    print(f"   dim={dim}, layers={n_layers}, heads={n_heads}, kv={n_kv}")
    print(f"   vocab={vocab_size}")
    print(f"   Total tensors in state_dict: {len(state_dict)}")

    # GGUF Writer - architecture="llama"
    writer = gguf.GGUFWriter(output_path, "llama")
    
    # ─── Metadata ───
    writer.add_context_length(2048)
    writer.add_embedding_length(dim)
    writer.add_block_count(n_layers)
    writer.add_feed_forward_length(dim * 4)  # SwiGLU
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv)
    writer.add_rope_dimension_count(head_dim)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_file_type(gguf.GGMLQuantizationType.F32)
    
    # Tokenizer
    writer.add_tokenizer_model("bpe")
    writer.add_tokenizer_pre("default")
    
    # ─── Tensor mapping ───
    # MATERIA -> Llama compatible
    tensor_map = {
        'tok_emb.weight': 'token_embd.weight',
        'head.weight': 'output.weight',
        'norm.weight': 'output_norm.weight',
    }
    
    layer_map = {
        'attention_norm.weight': 'blk.{i}.attn_norm.weight',
        'ffn_norm.weight': 'blk.{i}.ffn_norm.weight',
        'attention.wq.weight': 'blk.{i}.attn_q.weight',
        'attention.wk.weight': 'blk.{i}.attn_k.weight',
        'attention.wv.weight': 'blk.{i}.attn_v.weight',
        'attention.wo.weight': 'blk.{i}.attn_output.weight',
        'ffn.w1.weight': 'blk.{i}.ffn_gate.weight',  # SwiGLU gate
        'ffn.w2.weight': 'blk.{i}.ffn_down.weight',  # SwiGLU down
        'ffn.w3.weight': 'blk.{i}.ffn_up.weight',    # SwiGLU up
    }
    
    written = 0
    skipped = 0
    
    for k, v in state_dict.items():
        # Check direct mapping
        gguf_name = tensor_map.get(k)
        
        # Check layer mapping
        if gguf_name is None:
            for lid in range(n_layers):
                prefix = f'layers.{lid}.'
                if k.startswith(prefix):
                    suffix = k[len(prefix):]
                    if suffix in layer_map:
                        gguf_name = layer_map[suffix].replace('{i}', str(lid))
                    break
        
        if gguf_name:
            # Convert to float32
            arr = np.array(v, dtype=np.float32)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            writer.add_tensor(gguf_name, arr)
            written += 1
        else:
            skipped += 1
    
    print(f"   Tensores escritos: {written}")
    print(f"   Tensores omitidos (MATERIA-specific): {skipped}")
    
    # ─── Write GGUF ───
    print(f"💾 Escribiendo GGUF compatible...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    file_size = os.path.getsize(output_path)
    print(f"✅ GGUF Llama-compatible creado: {output_path}")
    print(f"   Tamaño: {file_size/1024/1024/1024:.2f} GB")
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Uso: python3 convert_to_gguf_compat.py <input.basemateria> <output.gguf>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"❌ No existe: {input_path}")
        sys.exit(1)
    
    result = convert_to_compatible_gguf(input_path, output_path)
    print(f"\n🎯 Archivo listo: {result}")
