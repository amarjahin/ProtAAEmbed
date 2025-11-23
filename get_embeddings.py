import torch
import esm

# 1. Load model + alphabet
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()

# This is a list where index i stores the token string
tokens = alphabet.all_toks

for i, tok in enumerate(tokens):
    print(i, tok)

batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout

# 2. Prepare sequences
# Format: list of (sequence_id, sequence_string)
amino_acids = [
    "A", "R", "N", "D", "C",
    "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V",
]
data = [
    # ("protein1", "MKTFFVAGLALAVATASA"),
    ("protein2", "ARNDCQEGHILMFPSTWYV"),
]

batch_labels, batch_strs, batch_tokens = batch_converter(data)
token_embedding_matrix = model.embed_tokens.weight  # nn.Embedding.weight
print(token_embedding_matrix.shape)
aa_tokens = [tok for tok in alphabet.all_toks if tok.isalpha()]
aa_indices = [alphabet.get_idx(tok) for tok in aa_tokens]

aa_embeddings = token_embedding_matrix[aa_indices]  # shape: (20, d_model)
print(aa_embeddings.shape)


# print(batch_tokens)
# batch_tokens: [batch_size, seq_len] of token indices

# 3. Run the model with repr_layers to get token embeddings
# with torch.no_grad():
#     out = model(
#         batch_tokens,
#         repr_layers=[6],     # final layer index for esm2_t6_8M (6 layers total)
#         return_contacts=False,
#     )

# token_reprs = out["representations"][6]  # [batch_size, seq_len, d_model]
# # Note: includes BOS/EOS/PAD tokens – you usually want to strip those

# # 4. Strip BOS/EOS to get pure amino acid embeddings
# # alphabet.get_idx("<cls>"), "<eos>", "<pad>" if you need indices
# sequence_reprs = []
# for i, seq in enumerate(batch_strs):
#     # seq length in amino acids
#     L = len(seq)
#     # In ESM, tokens are: [CLS] A1 A2 ... AL [EOS] (padded afterwards)
#     # So residues correspond to positions 1..L (inclusive)
#     aa_repr = token_reprs[i, 1 : 1 + L, :]   # shape [L, d_model]
#     sequence_reprs.append(aa_repr)

# # representation for each amino acid 

# for i , seq in enumerate(batch_strs):
#     for j , aa in enumerate(seq):
#         print(aa, sequence_reprs[i][j].shape)