import torch
from unet import UNetModelWrapper


from peft import LoraConfig, TaskType, get_peft_model


checkpoint_path = "/projects/illinois/eng/cs/arindamb/gudrunt2/constrained-lora/checkpoints/ot-cfm/otcfm_cifar10_weights_step_400000.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print(checkpoint.keys())


net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        #channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
)

net_model.load_state_dict(checkpoint["net_model"])
net_model.eval()

# show model size
model_size = 0
for param in net_model.parameters():
    model_size += param.data.nelement()
print("Model params: %.2f M" % (model_size / 1024 / 1024))



#adapt model for LoRA

peft_config = LoraConfig(
        r=16, #rank, need to tune this?
        lora_alpha=32,  #scale ????
        target_modules='all-linear', #?????????????
        lora_dropout=0.05,
        bias="none"
    )

lora_model = get_peft_model(net_model, peft_config)
lora_model.eval()

lora_size = 0
for param in lora_model.parameters():
    lora_size += param.data.nelement()
print("LoRA Model params: %.2f M" % (lora_size / 1024 / 1024))

lora_model.print_trainable_parameters()





dummy = torch.randn(1, 3, 128, 128)
t = torch.tensor([0.1])

with torch.no_grad():
    output = net_model(t, dummy)

print(output.shape)

with torch.no_grad():
    output = lora_model(t, dummy)

print(output.shape)


