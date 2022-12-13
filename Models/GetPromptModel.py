"""
build_promptmodel   Script  ver： Aug 31th 15:20

"""

try:
    from .VPT_structure import *
except:
    from Models.VPT_structure import *


def build_promptmodel(num_classes=1000, edge_size=224, model_idx='ViT', patch_size=16,
                      Prompt_Token_num=20, VPT_type="Deep", prompt_state_dict=None, base_state_dict='timm'):
    # VPT_type = "Deep" / "Shallow"

    if model_idx[0:3] == 'ViT':

        if base_state_dict is None:
            basic_state_dict = None

        elif type(base_state_dict) == str:
            if base_state_dict == 'timm':
                # ViT_Prompt
                import timm
                # from pprint import pprint
                # model_names = timm.list_models('*vit*')
                # pprint(model_names)

                basic_model = timm.create_model('vit_base_patch' + str(patch_size) + '_' + str(edge_size),
                                                pretrained=True)
                basic_state_dict = basic_model.state_dict()
                print('in prompt model building, timm ViT loaded for base_state_dict')

            else:
                basic_state_dict = None
                print('in prompt model building, no vaild str for base_state_dict')

        else:  # state dict: collections.OrderedDict
            basic_state_dict = base_state_dict
            print('in prompt model building, a .pth base_state_dict loaded')

        model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                        VPT_type=VPT_type, basic_state_dict=basic_state_dict)

        model.New_CLS_head(num_classes)

        if prompt_state_dict is not None:
            try:
                model.load_prompt(prompt_state_dict)
            except:
                print('erro in .pth prompt_state_dict')
            else:
                print('in prompt model building, a .pth prompt_state_dict loaded')

        model.Freeze()
    else:
        print("The model is not difined in the Prompt script！！")
        return -1

    try:
        img = torch.randn(1, 3, edge_size, edge_size)
        preds = model(img)  # (1, class_number)
        # print('test model output：', preds)
        print('model forward cheacked')

    except:
        print("Problem exist in the model defining process！！")
        return -1
    else:
        print('model is ready now!')
        return model
