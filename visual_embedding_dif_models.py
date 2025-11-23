import torch
import esm
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from transformers import AutoModel, AutoTokenizer

# SeqVec support removed due to Python 3.14 compatibility issues
# To add SeqVec back, use Python 3.12 or 3.13 and install bio-embeddings or allennlp

# -----------------------------
# 1. MODELS TO RUN
# -----------------------------
models = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "protbert",  # ProtBERT model
    # "seqvec",  # SeqVec model - skipped due to Python 3.14 compatibility issues
]

# -----------------------------
# 2. Chemical property groups
# -----------------------------
nonpolar = {"A", "V", "L", "I", "M", "F", "W", "P", "G"}
polar_uncharged = {"S", "T", "N", "Q", "C", "Y"}
positive = {"K", "R", "H"}
negative = {"D", "E"}

def aa_property(aa: str) -> str:
    if aa in nonpolar:
        return "nonpolar"
    elif aa in polar_uncharged:
        return "polar_uncharged"
    elif aa in positive:
        return "positive"
    elif aa in negative:
        return "negative"
    else:
        return "other"

color_map = {
    "nonpolar": "tab:blue",
    "polar_uncharged": "tab:green",
    "positive": "tab:red",
    "negative": "tab:purple",
    "other": "tab:gray",
}

order = ["nonpolar", "polar_uncharged", "positive", "negative", "other"]

# -----------------------------
# 3. Helper function to extract embeddings
# -----------------------------
def get_aa_embeddings(model_name):
    """Extract amino acid embeddings from ESM or ProtBERT model."""
    if model_name == "protbert":
        # Load ProtBERT model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
        model = AutoModel.from_pretrained("Rostlab/prot_bert")
        model.eval()
        
        # Standard 20 amino acids
        aa_tokens = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
                     "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        
        # Get embedding layer
        embedding_layer = model.embeddings.word_embeddings
        
        # Get token IDs for each amino acid
        # ProtBERT tokenizer works with amino acids directly
        aa_embeddings = []
        valid_aa_tokens = []
        
        for aa in aa_tokens:
            # Tokenize the amino acid (ProtBERT tokenizer handles single AAs)
            # Use convert_tokens_to_ids for direct token ID lookup
            try:
                token_id = tokenizer.convert_tokens_to_ids(aa)
                if token_id != tokenizer.unk_token_id:  # Make sure it's not unknown
                    valid_aa_tokens.append(aa)
                    aa_embeddings.append(embedding_layer.weight[token_id].detach())
                else:
                    # Fallback: try encoding
                    token_ids = tokenizer.encode(aa, add_special_tokens=False)
                    if len(token_ids) > 0:
                        valid_aa_tokens.append(aa)
                        aa_embeddings.append(embedding_layer.weight[token_ids[0]].detach())
            except:
                # Fallback: encode the amino acid
                token_ids = tokenizer.encode(aa, add_special_tokens=False)
                if len(token_ids) > 0:
                    valid_aa_tokens.append(aa)
                    aa_embeddings.append(embedding_layer.weight[token_ids[0]].detach())
        
        if len(aa_embeddings) == 0:
            raise ValueError("Could not extract any amino acid embeddings from ProtBERT")
        
        aa_embeddings = torch.stack(aa_embeddings)
        X = aa_embeddings.numpy()
        return X, valid_aa_tokens
    
    else:
        # ESM model
        model, alphabet = esm.pretrained.__dict__[model_name]()
        model.eval()
        
        # Extract raw token embeddings
        token_embedding_matrix = model.embed_tokens.weight.detach()
        
        aa_tokens = [tok for tok in alphabet.all_toks if tok.isalpha()]
        aa_indices = [alphabet.get_idx(tok) for tok in aa_tokens]
        aa_embeddings = token_embedding_matrix[aa_indices]
        X = aa_embeddings.numpy()
        return X, aa_tokens

# -----------------------------
# 4. Create subplots and visualize
# -----------------------------
n_models = len(models)
n_rows, n_cols = 2, 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
axes = axes.flatten()

for ax, model_name in zip(axes, models):
    try:
        # ---- Extract embeddings ----
        X, aa_tokens = get_aa_embeddings(model_name)
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        # Create empty plot with error message
        ax.text(0.5, 0.5, f"Error:\n{str(e)[:50]}...", 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8, color='red')
        ax.set_title(model_name)
        continue

    # ---- t-SNE ----
    tsne = TSNE(
        n_components=2,
        perplexity=8,
        learning_rate="auto",
        init="random",
        random_state=0,
    )
    X_2d = tsne.fit_transform(X)

    # ---- Plot ----
    colors = [color_map[aa_property(aa)] for aa in aa_tokens]
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors)
    for i, aa in enumerate(aa_tokens):
        ax.annotate(aa, (X_2d[i, 0], X_2d[i, 1]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_title("ProtBERT" if model_name == "protbert" else model_name)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

# -----------------------------
# 5. Place legend in the last subplot
# -----------------------------
legend_ax = axes[n_models]  # Use the 6th subplot for legend
legend_ax.axis('off')  # Remove axes

legend_handles = [
    Patch(facecolor=color_map[prop], label=prop.replace("_", " "))
    for prop in order
]

legend_ax.legend(
    handles=legend_handles,
    title="Chemical property",
    loc="center",
    frameon=True,
)

plt.tight_layout()
plt.show()