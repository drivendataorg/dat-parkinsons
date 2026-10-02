import torch
from torch import nn
from . import resnet
from . import resnext
import torch.serialization

__name__='models.resnet'
SAFE_GLOBALS = [__name__]

RESNEXT_MODELS = {
    50: resnext.resnext50,
    101: resnext.resnext101,
    152: resnext.resnext152,
    200: resnext.resnext200,
    269: resnext.resnext269,
}

RESNET_MODELS = {
    10: resnet.resnet10,
    18: resnet.resnet18,
    34: resnet.resnet34,
    50: resnet.resnet50,
    101: resnet.resnet101,
    152: resnet.resnet152,
    200: resnet.resnet200,
}

MODEL_REGISTRY = {
    "resnext": (RESNEXT_MODELS, "resnext"),
    "resnet": (RESNET_MODELS, "resnet"),
}

# ===================================================================================================

def generate_model(opt):
    backbone = opt.medicalnet_backbone
    depth = opt.medicalnet_depth
    assert depth is not None
    assert backbone in MODEL_REGISTRY.keys()

    resnext_args = dict(
        num_classes=opt.n_classes,
        shortcut_type=opt.resnet_shortcut,
        cardinality=opt.resnext_cardinality,
        sample_size=opt.sample_size,
        sample_duration=opt.sample_duration,
    )

    resnet_args = dict(
        sample_input_W=opt.input_W,
        sample_input_H=opt.input_H,
        sample_input_D=opt.input_D,
        shortcut_type=opt.resnet_shortcut,
        no_cuda=opt.no_cuda,
        num_seg_classes=opt.n_seg_classes,
    )
        
    constructors, family = MODEL_REGISTRY[backbone]
    try:
        constructor = constructors[depth]
    except KeyError:
        raise ValueError(
            f"Depth {depth} is unsupported "
            f"for {backbone}"
        )

    args = resnet_args if family == "resnet" else resnext_args
    model = constructor(**args)


    if not opt.no_cuda:
        if len(opt.gpu_id) > 1:
            model = model.cuda() 
            # model = nn.DataParallel(model, device_ids=opt.gpu_id)
            net_dict = model.state_dict() 
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_id[0])
            model = model.cuda() 
            # model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()


    if opt.phase != 'test' and opt.pretrain_path:
        print(f'正在加载预训练模型 {opt.pretrain_path}')
        try:
            with torch.serialization.safe_globals(SAFE_GLOBALS):
                pretrain = torch.load(
                    opt.pretrain_path,
                    weights_only=True,
                    map_location='cpu'  
                )
        
        except FileNotFoundError:
            raise FileNotFoundError(f"预训练模型文件不存在: {opt.pretrain_path}")
        except RuntimeError as e:
            if "is not in the safe globals list" in str(e):
                raise RuntimeError(
                    f"加载模型时安全检查失败！请将相关模块添加到白名单。错误详情：{e}\n"
                    f"当前白名单模块：{SAFE_GLOBALS}"
                ) from e
            else:
                raise RuntimeError(f"加载预训练模型失败：{e}") from e
        except Exception as e:
            raise Exception(f"加载模型时发生未知错误：{e}") from e
        
        
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
         
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
        print("Loaded from", opt.pretrain_path)

        new_parameters = [] 
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters, 
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()
