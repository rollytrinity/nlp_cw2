import torch

from model import GPT, GPTConfig
# Set random seed for reproducibility
# DO NOT CHANGE!
torch.manual_seed(42)

model_config = GPTConfig(
    vocab_size=100,
    block_size=4,
    n_layer=2,
    n_head=6,
    n_embed=12,
    dropout=0.0,
)

model = GPT(model_config)

input_embeds = torch.randn(2, model_config.block_size, model_config.n_embed)
with torch.inference_mode():
    attention_output = model.blocks[0](input_embeds, output_attentions=True).attentions
    expected_output = torch.load("sanity_check.pt")

# check shapes
shape_check = attention_output.shape == expected_output.shape
if not shape_check:
    print(
        f"Attention shape does not match expected shape. Expected {expected_output['attention'].shape} but got {attention_output.shape}."
    )

# check values
attention_check = torch.allclose(attention_output, expected_output, atol=1e-6)
if not attention_check:
    print("Attention values do not match expected output.")

passed = shape_check and attention_check
print("All checks passed!" if passed else "Some checks failed, please see above for details.")