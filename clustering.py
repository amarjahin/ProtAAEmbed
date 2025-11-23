import torch
import esm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
from transformers import AutoModel, AutoTokenizer

# -------------------------------------------------------
# 1. Models to compare
# -------------------------------------------------------
models = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    # "esm2_t36_3B_UR50D",
    # "esm2_t48_15B_UR50D",
    "protbert",  # ProtBERT model
]

# -------------------------------------------------------
# 2. Chemical properties
# -------------------------------------------------------
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

# -------------------------------------------------------
# 3. Helper function to extract embeddings
# -------------------------------------------------------
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
        aa_embeddings = []
        valid_aa_tokens = []
        
        for aa in aa_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(aa)
                if token_id != tokenizer.unk_token_id:
                    valid_aa_tokens.append(aa)
                    aa_embeddings.append(embedding_layer.weight[token_id].detach())
                else:
                    token_ids = tokenizer.encode(aa, add_special_tokens=False)
                    if len(token_ids) > 0:
                        valid_aa_tokens.append(aa)
                        aa_embeddings.append(embedding_layer.weight[token_ids[0]].detach())
            except:
                token_ids = tokenizer.encode(aa, add_special_tokens=False)
                if len(token_ids) > 0:
                    valid_aa_tokens.append(aa)
                    aa_embeddings.append(embedding_layer.weight[token_ids[0]].detach())
        
        if len(aa_embeddings) == 0:
            raise ValueError("Could not extract any amino acid embeddings from ProtBERT")
        
        aa_embeddings = torch.stack(aa_embeddings)
        return aa_embeddings.numpy(), valid_aa_tokens
    
    else:
        # ESM model
        model, alphabet = esm.pretrained.__dict__[model_name]()
        model.eval()
        
        # Extract raw token embeddings
        emb = model.embed_tokens.weight.detach()
        
        # Extract amino acid tokens
        aa_tokens = [tok for tok in alphabet.all_toks if tok.isalpha()]
        aa_indices = [alphabet.get_idx(tok) for tok in aa_tokens]
        aa_embeddings = emb[aa_indices].numpy()
        
        return aa_embeddings, aa_tokens

# -------------------------------------------------------
# 4. Purity metric
# -------------------------------------------------------
def purity_score(y_true_str, y_pred):
    total = 0
    y_true_str = np.asarray(y_true_str)
    for k in np.unique(y_pred):
        mask_k = (y_pred == k)
        majority_count = Counter(y_true_str[mask_k]).most_common(1)[0][1]
        total += majority_count
    return total / len(y_true_str)

# -------------------------------------------------------
# 5. Run models and compute metrics
# -------------------------------------------------------
results = []

for model_name in models:
    print(f"\n=== Running {model_name} ===")
    
    # Extract embeddings
    aa_embeddings, aa_tokens = get_aa_embeddings(model_name)

    # t-SNE 2D projection
    tsne = TSNE(
        n_components=2,
        perplexity=8,
        learning_rate="auto",
        init="random",
        random_state=0,
    )
    X_2d = tsne.fit_transform(aa_embeddings)

    # Assign chemical properties as ground-truth labels
    properties = np.array([aa_property(a) for a in aa_tokens])

    # String → integer for metrics
    unique_props = sorted(set(properties))
    prop2id = {p: i for i, p in enumerate(unique_props)}
    y_true_int = np.array([prop2id[p] for p in properties])

    # KMeans: 4 clusters for ProtBERT, 5 for ESM models
    n_clusters = 4 if model_name == "protbert" else 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(X_2d)

    # Metrics
    purity = purity_score(properties, cluster_labels)
    ari = adjusted_rand_score(y_true_int, cluster_labels)
    nmi = normalized_mutual_info_score(y_true_int, cluster_labels)

    display_name = "ProtBERT" if model_name == "protbert" else model_name
    print(f"Purity: {purity:.3f}   ARI: {ari:.3f}   NMI: {nmi:.3f}")
    results.append((display_name, purity, ari, nmi))

# -------------------------------------------------------
# 5. Print results nicely
# -------------------------------------------------------
print("\n================ RESULTS ================\n")
print(f"{'Model':25s}  {'Purity':>6s}  {'ARI':>6s}  {'NMI':>6s}")
print("-" * 50)
for name, purity, ari, nmi in results:
    print(f"{name:25s}  {purity:6.3f}  {ari:6.3f}  {nmi:6.3f}")