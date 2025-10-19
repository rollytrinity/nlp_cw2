import torch

from nmt_model import NMT

# Set random seed for reproducibility
# DO NOT CHANGE!
torch.manual_seed(42)

batch_size = 2
src_len = 3
src_vocab_size = 5
tgt_vocab_size = 5
src_pad_token_idx = 0
tgt_pad_token_idx = 0

model_config = {
    # The size of the embedding vectors
    "embed_size": 4,
    # The size of the encoder's LSTM hidden states
    "hidden_size": 4,
    # The dropout rate
    "dropout_rate": 0.2
}

model = NMT(
    embed_size=model_config["embed_size"],
    hidden_size=model_config["hidden_size"],
    dropout_rate=model_config["dropout_rate"],
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    src_pad_token_idx=src_pad_token_idx,
    tgt_pad_token_idx=tgt_pad_token_idx,
)


Ybar_t = torch.randn(batch_size, model_config["embed_size"] + model_config["hidden_size"])
dec_state = (
    torch.randn(batch_size, model_config["hidden_size"]),
    torch.randn(batch_size, model_config["hidden_size"]),
)
enc_hiddens = torch.randn(batch_size, src_len, 2 * model_config["hidden_size"])
enc_hiddens_proj = torch.randn(batch_size, src_len, model_config["hidden_size"])

masks = []
padded = 0
for _ in range(batch_size):
    mask = [0] * (src_len - padded) + [1] * padded
    masks.append(mask)
    padded += 1

enc_masks = torch.tensor(masks)
with torch.inference_mode():
    dec_state, o_t, alpha_t = model.step(
        Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj , enc_masks
    )

expected_outputs = torch.load("sanity_check.pt")

dec_hidden_check = torch.allclose(dec_state[0], expected_outputs["dec_state"][0], atol=1e-6)
if not dec_hidden_check:
    print("Decoder hidden state does not match expected output.")

dec_cell_check = torch.allclose(dec_state[1], expected_outputs["dec_state"][1], atol=1e-6)
if not dec_cell_check:
    print("Decoder cell state does not match expected output.")

o_t_check = torch.allclose(o_t, expected_outputs["o_t"], atol=1e-6)
if not o_t_check:
    print("Output tensor does not match expected output.")

alpha_t_check = torch.allclose(alpha_t, expected_outputs["alpha_t"], atol=1e-6)
if not alpha_t_check:
    print("Attention weights do not match expected output.")

passed = dec_hidden_check and dec_cell_check and o_t_check and alpha_t_check
print("All checks passed!" if passed else "Some checks failed, please see above for details.")