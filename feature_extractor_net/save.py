import torch
from model import Net
def cvt_model():
    print("===> Loading model")
    model = Net(reid=True)
    modelname = 'feature_extractor_net/checkpoint/ckpt.t7'
    checkpoint = torch.load(modelname)
    model.load_state_dict(checkpoint['net_dict'])  # 从字典中依次读取，具体值查看字典更改
    print('===> Load last checkpoint data')

    # 模型转换，Torch Script
    model.eval()
    example = torch.rand(4,3,128,64)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("feature_extractor_net/checkpoint/new_model.pt")
    print("Export of model.pt complete!")
cvt_model()