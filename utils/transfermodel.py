"""
transfer Pre-Training checkpoints    Script  ver： Aug 15th 19:00

write a model based on the weight of a checkpoint file
EG: create a vit-base based on PuzzleTuning SAE

"""
import os
import torch
import torch.nn as nn

from Hybrid import getmodel
from PromptModels import GetPromptModel
from PromptModels import models_mae, SAE


# Transfer pretrained MSHT checkpoints to normal model state_dict
def transfer_model_encoder(check_point_path, save_model_path, model_idx='ViT', prompt_mode=None,
                           Prompt_Token_num=20, edge_size=384):
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    if prompt_mode == "Deep" or prompt_mode == "Shallow":
        model = GetPromptModel.build_promptmodel(edge_size=edge_size, model_idx=model_idx, patch_size=16,
                                                 Prompt_Token_num=Prompt_Token_num, VPT_type=prompt_mode)
    # elif prompt_mode == "Other":
    else:
        model = getmodel.get_model(model_idx=model_idx, pretrained_backbone=False, edge_size=edge_size)
    '''
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    TempBest_state = {'model': best_model_wts, 'epoch': best_epoch_idx}
    '''
    state = torch.load(check_point_path)

    transfer_name = os.path.splitext(os.path.split(check_point_path)[1])[0] + '_of_'

    try:
        model_state = state['model']
        try:
            print("checkpoint epoch", state['epoch'])
            if prompt_mode is not None:
                save_model_path = os.path.join(save_model_path, transfer_name +
                                               model_idx + '_E_' + str(state['epoch']) + '_promptstate' + '.pth')
            else:
                save_model_path = os.path.join(save_model_path, transfer_name +
                                               model_idx + '_E_' + str(state['epoch']) + '_transfer' + '.pth')

        except:
            print("no 'epoch' in state")
            if prompt_mode is not None:
                save_model_path = os.path.join(save_model_path, transfer_name + model_idx + '_promptstate' + '.pth')
            else:
                save_model_path = os.path.join(save_model_path, transfer_name + model_idx + '_transfer' + '.pth')
    except:
        print("not a checkpoint state (no 'model' in state)")
        model_state = state
        if prompt_mode is not None:
            save_model_path = os.path.join(save_model_path, transfer_name + model_idx + '_promptstate' + '.pth')
        else:
            save_model_path = os.path.join(save_model_path, transfer_name + model_idx + '_transfer' + '.pth')

    try:
        model.load_state_dict(model_state)
        print("model loaded")
        print("model :", model_idx)
        gpu_use = 0
    except:
        try:
            model = nn.DataParallel(model)
            model.load_state_dict(model_state, False)
            print("DataParallel model loaded")
            print("model :", model_idx)
            gpu_use = -1
        except:
            print("model loading erro!!")
            gpu_use = -2

    if gpu_use == -1:
        # print(model)
        if prompt_mode is not None:
            prompt_state_dict = model.module.obtain_prompt()
            # fixme maybe bug at DP module.obtain_prompt, just model.obtain_prompt is enough
            print('prompt obtained')
            torch.save(prompt_state_dict, save_model_path)
        else:
            torch.save(model.module.state_dict(), save_model_path)
        print('model trained by multi-GPUs has its single GPU copy saved at ', save_model_path)

    elif gpu_use == 0:
        if prompt_mode is not None:
            prompt_state_dict = model.obtain_prompt()
            print('prompt obtained')
            torch.save(prompt_state_dict, save_model_path)
        else:
            torch.save(model.state_dict(), save_model_path)
        print('model trained by a single GPU has been saved at ', save_model_path)
    else:
        print('erro')


def transfer_model_decoder(check_point_path, save_model_path,
                           model_idx='sae_vit_base_patch16_decoder', dec_idx='swin_unet',
                           prompt_mode=None, Prompt_Token_num=20, edge_size=384):

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    state = torch.load(check_point_path)

    transfer_name = os.path.splitext(os.path.split(check_point_path)[1])[0] + '_of_'

    model = SAE.__dict__[model_idx](img_size=edge_size, prompt_mode=prompt_mode, Prompt_Token_num=Prompt_Token_num,
                                    basic_state_dict=None, dec_idx=dec_idx)

    try:
        model_state = state['model']
        try:
            print("checkpoint epoch", state['epoch'])
            save_model_path = os.path.join(save_model_path, transfer_name + model_idx + '_E_'
                                           + str(state['epoch']) + '_transfer_Decoder_'+dec_idx + '.pth')


        except:
            print("no 'epoch' in state")
            save_model_path = os.path.join(save_model_path, transfer_name + model_idx
                                           + '_transfer_Decoder_'+dec_idx + '.pth')
    except:
        print("not a checkpoint state (no 'model' in state)")
        model_state = state
        save_model_path = os.path.join(save_model_path, transfer_name + model_idx
                                       + '_transfer_Decoder_'+dec_idx + '.pth')

    try:
        model.load_state_dict(model_state)
        print("model loaded")
        print("model :", model_idx)
        gpu_use = 0
    except:
        try:
            model = nn.DataParallel(model)
            model.load_state_dict(model_state, False)
            print("DataParallel model loaded")
            print("model :", model_idx)
            gpu_use = -1
        except:
            print("model loading erro!!")
            gpu_use = -2

    else:
        model = model.decoder

    if gpu_use == -1:
        torch.save(model.module.decoder.state_dict(), save_model_path)
        print('model trained by multi-GPUs has its single GPU copy saved at ', save_model_path)

    elif gpu_use == 0:
        torch.save(model.state_dict(), save_model_path)
        print('model trained by a single GPU has been saved at ', save_model_path)
    else:
        print('erro')


if __name__ == '__main__':
    # fixme: now need a CUDA device as the model is save as a CUDA model!

    check_point_path = '/root/autodl-tmp/PuzzleTuning_sae_vit_base_patch16_decoder_Deep/checkpoints/PuzzleTuning_sae_vit_base_patch16_decoder_Deepcheckpoint-399.pth'
    save_model_path = '/root/autodl-tmp/output_models'

    # os.path.split(check_point_path)[1].split('.')[0].split('_')[1]  # 'PT_' + model_idx + _xxxx + '.pth'

    transfer_model_encoder(check_point_path, save_model_path, model_idx='ViT', edge_size=224, prompt_mode='Deep')

    transfer_model_decoder(check_point_path, save_model_path,
                           model_idx='sae_vit_base_patch16_decoder',
                           edge_size=224,
                           prompt_mode='Deep')
