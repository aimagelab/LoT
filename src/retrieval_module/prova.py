import argparse
import os
import PIL
import torch

from retrieval_module.retrieval import InferenceModelQwen2_5_VL, get_args, InferenceModelInternVL, InferenceModelQwen3_VL, InferenceModelQwen2VL

args = get_args()

args.model_name = "OpenGVLab/InternVL3_5-38B-hf" # "Qwen/Qwen2.5-VL-3B-Instruct"

if "Intern" in args.model_name:
    model = InferenceModelInternVL(args)
elif "Qwen2-VL" in args.model_name:
    model = InferenceModelQwen2VL(args)
elif "Qwen2.5-Vl" in args.model_name:
    model = InferenceModelQwen2_5_VL(args)
elif "Qwen3-VL" in args.model_name:
    model = InferenceModelQwen3_VL(args)
else:
    raise ValueError("Model name must contain either 'Intern' or 'Qwen'")

image = PIL.Image.open("/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/dataset/iNaturalist_2021/val/09999_Plantae_Tracheophyta_Polypodiopsida_Schizaeales_Lygodiaceae_Lygodium_japonicum/07298481-732e-42a2-a8bd-9f07ebb51a60.jpg").convert("RGB")
question = "Who discovered this plant?"
# spans = [(307, 308)]

accumulated_activations = []

def hook_fn(module, input, output):
    accumulated_activations.append(output[0].detach().cpu())


target_layer_idx = 2
model.model.language_model.layers[target_layer_idx].register_forward_hook(hook_fn)

inputs = model.get_inputs(question, image)
inputs = inputs.to(model.model.device)

processor = model.processor
token_vis_start = "<img>" if "Intern" in args.model_name else "<|vision_start|>"
token_vis_end = "</img>" if "Intern" in args.model_name else "<|vision_end|>"
vision_start_id = processor.tokenizer.convert_tokens_to_ids(token_vis_start)
vision_end_id = processor.tokenizer.convert_tokens_to_ids(token_vis_end)
input_ids_tensor = inputs.input_ids[0]
vision_start_idx = (input_ids_tensor == vision_start_id).nonzero(as_tuple=True)[0].item()
vision_end_idx = (input_ids_tensor == vision_end_id).nonzero(as_tuple=True)[0].item()

output = model.model(**inputs)

print(output)

hidden_states = accumulated_activations[0][0] if "Qwen2" in args.model_name else accumulated_activations[0] # [seq_len, hidden_dim]

# Calcola la media e trova i picchi
# Cerchiamo dimensioni dove il valore è enormemente più alto della media
mean_activation = hidden_states.abs().mean(dim=0) # Media lungo la sequenza per ogni dimensione
values, indices = torch.topk(mean_activation, k=10) # Prendi le top 10 dimensioni

print(f"\n--- Dimensioni Critiche nel Layer {target_layer_idx} ---")
for i, (val, idx) in enumerate(zip(values, indices)):
    print(f"Dimensione {idx.item()}: Valore Medio Assoluto = {val.item():.2f}")

# Controllo aggiuntivo sul token BOS (indice 0 solitamente)
bos_state = hidden_states[0] # Stato del primo token
bos_values, bos_indices = torch.topk(bos_state.abs(), k=5)
print(f"\n--- Attivazioni nel Token BOS ---")
for val, idx in zip(bos_values, bos_indices):
    print(f"Dimensione {idx.item()}: Valore = {val.item():.2f}")

vision_states = hidden_states[vision_start_idx + 1: vision_end_idx]
# Calcola la media e trova i picchi nelle attivazioni visive
mean_vision_activation = vision_states.abs().mean(dim=0)
vision_values, vision_indices = torch.topk(mean_vision_activation, k=10)
print(f"\n--- Dimensioni Critiche nelle Attivazioni Visive ---")
for i, (val, idx) in enumerate(zip(vision_values, vision_indices)):
    print(f"Dimensione {idx.item()}: Valore Medio Assoluto = {val.item():.2f}")


# --- Plot di tutte le attivazioni BOS ---
import matplotlib.pyplot as plt
import numpy as np

rms = torch.sqrt(bos_state.pow(2).mean())
bos_state = bos_state / (rms + 1e-6)

all_bos_vals = bos_state.abs().to(dtype=torch.float32).cpu().numpy()
dim_indices = np.arange(len(all_bos_vals))

# Prepara i top-k per evidenziare
topk_indices = set(bos_indices.cpu().numpy())

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(dim_indices, all_bos_vals, color=["#00bfff" if i in topk_indices else "#cccccc" for i in dim_indices])

# Evidenzia i top-k in grassetto
for i, bar in enumerate(bars):
    if i in topk_indices:
        bar.set_linewidth(2.5)
        bar.set_edgecolor('black')

# Traccia linee verticali tratteggiate per i top-k
for idx in topk_indices:
    ax.axvline(x=idx, color='gray', linestyle='--', linewidth=1, alpha=0.4)

ax.set_title("Attivazioni |BOS| per dimensione (top-k evidenziati)", fontsize=16)
ax.set_xlabel("Dimensione", fontsize=13)
ax.set_ylabel("Valore assoluto attivazione", fontsize=13)
ax.set_xlim([0, len(all_bos_vals)])

# Evidenzia i tick dell'asse x corrispondenti ai top-k in grassetto
xticks = ax.get_xticks()
xticklabels = ax.get_xticklabels()
for i, (tick, label) in enumerate(zip(xticks, xticklabels)):
    if int(tick) in topk_indices:
        label.set_fontweight('bold')
        label.set_color('#00bfff')

plt.tight_layout()
plt.savefig("bos_activations.png")
plt.show()
