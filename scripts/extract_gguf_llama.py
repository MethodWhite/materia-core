"""
Extract Llama-compatible tensors from the full MATERIA GGUF.
Creates a standard GGUF that LM Studio / Ollama / llama.cpp can load.
"""
import os, sys
import numpy as np
import gguf

def extract_llama_compatible(input_gguf, output_gguf):
    print(f"📖 Reading GGUF: {input_gguf}")
    reader = gguf.GGUFReader(input_gguf)
    
    # Get metadata
    arch = reader.fields.get('general.architecture')
    if arch:
        print(f"   Architecture: {arch.parts[-1]}")
    
    dim = reader.fields.get('llama.embedding_length')
    n_layers = reader.fields.get('llama.block_count')
    n_heads = reader.fields.get('llama.head_count')
    n_kv = reader.fields.get('llama.head_count_kv')
    ffn_dim = reader.fields.get('llama.feed_forward_length')
    
    dim_val = int(dim.parts[-1]) if dim else 1792
    layers_val = int(n_layers.parts[-1]) if n_layers else 24
    heads_val = int(n_heads.parts[-1]) if n_heads else 24
    kv_val = int(n_kv.parts[-1]) if n_kv else 6
    
    print(f"   dim={dim_val}, layers={layers_val}, heads={heads_val}, kv={kv_val}")
    
    # Tensor name mapping: from GGUF tensor names to Llama standard
    # GGUF stores tensors with their original names since we used custom mapping
    # We need to check what names the tensors have
    
    all_tensors = reader.tensors
    print(f"   Total tensors in GGUF: {len(all_tensors)}")
    
    # Print first 20 tensor names to understand naming
    for i, t in enumerate(all_tensors[:20]):
        print(f"     {t.name}: {t.shape}")
    
    # Create new GGUF with Llama architecture
    writer = gguf.GGUFWriter(output_gguf, "llama")
    
    # Metadata
    writer.add_context_length(2048)
    writer.add_embedding_length(dim_val)
    writer.add_block_count(layers_val)
    writer.add_feed_forward_length(dim_val * 4)
    writer.add_head_count(heads_val)
    writer.add_head_count_kv(kv_val)
    writer.add_rope_dimension_count(dim_val // heads_val)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_file_type(gguf.GGMLQuantizationType.F32)
    writer.add_tokenizer_model("bpe")
    writer.add_tokenizer_pre("default")
    
    # Llama-compatible tensor name mapping
    # MATERIA V4 TransformerBlock uses: attn_norm, ffn_norm, attn.wq/wk/wv/wo, ffn.gate/up/down
    tensor_name_map = {
        'tok_emb.weight': 'token_embd.weight',
        'norm.weight': 'output_norm.weight',
        'head.weight': 'output.weight',
    }
    
    layer_name_map = {
        'attn_norm.weight': 'blk.{i}.attn_norm.weight',
        'ffn_norm.weight': 'blk.{i}.ffn_norm.weight',
        'attn.wq.weight': 'blk.{i}.attn_q.weight',
        'attn.wk.weight': 'blk.{i}.attn_k.weight',
        'attn.wv.weight': 'blk.{i}.attn_v.weight',
        'attn.wo.weight': 'blk.{i}.attn_output.weight',
        'ffn.gate.weight': 'blk.{i}.ffn_gate.weight',
        'ffn.up.weight': 'blk.{i}.ffn_up.weight',
        'ffn.down.weight': 'blk.{i}.ffn_down.weight',
    }
    
    written = 0
    skipped = 0
    
    # FFN tensors were stored as materia.* because 'gate','up','down' triggered special classification
    # Need to extract them from materia.layers_X_ffn_gate_weight format
    ffn_map = {}
    for lid in range(layers_val):
        ffn_map[f'materia.layers_{lid}_ffn_gate_weight'] = f'blk.{lid}.ffn_gate.weight'
        ffn_map[f'materia.layers_{lid}_ffn_up_weight'] = f'blk.{lid}.ffn_up.weight'
        ffn_map[f'materia.layers_{lid}_ffn_down_weight'] = f'blk.{lid}.ffn_down.weight'
    
    for tensor in all_tensors:
        name = tensor.name
        data = tensor.data
        
        gguf_name = None
        
        # Direct match
        if name in tensor_name_map:
            gguf_name = tensor_name_map[name]
        elif name in ffn_map:
            gguf_name = ffn_map[name]
        else:
            # Layer match (attention tensors)
            for lid in range(layers_val):
                prefix = f'layers.{lid}.'
                if name.startswith(prefix):
                    suffix = name[len(prefix):]
                    if suffix in layer_name_map:
                        gguf_name = layer_name_map[suffix].replace('{i}', str(lid))
                    break
        
        if gguf_name:
            arr = np.array(data, dtype=np.float32)
            writer.add_tensor(gguf_name, arr)
            written += 1
            if written <= 5 or written % 20 == 0:
                print(f"   ✓ {name} -> {gguf_name} {arr.shape}")
        else:
            skipped += 1
    
    print(f"\n   Tensores escritos: {written}")
    print(f"   Tensores omitidos: {skipped}")
    
    # Write
    print(f"💾 Escribiendo GGUF Llama-compatible...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    size = os.path.getsize(output_gguf)
    print(f"✅ Creado: {output_gguf}")
    print(f"   Tamaño: {size/1024/1024/1024:.2f} GB")
    
    return output_gguf


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Uso: python3 extract_gguf_llama.py <input.gguf> <output.gguf>")
        sys.exit(1)
    
    result = extract_llama_compatible(sys.argv[1], sys.argv[2])
    print(f"\n🎯 Archivo listo: {result}")
