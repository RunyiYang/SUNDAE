import torch.nn as nn
import torch.nn.functional as F
from swin_t import SUNet
from IPython import embed

class SUNet_model(nn.Module):
    def __init__(self, config):
        super(SUNet_model, self).__init__()
        self.config = config
        self.swin_unet = SUNet(img_size=config['SWINUNET']['IMG_SIZE'],
                               patch_size=config['SWINUNET']['PATCH_SIZE'],
                               in_chans=3,
                               out_chans=3,
                               embed_dim=config['SWINUNET']['EMB_DIM'],
                               depths=config['SWINUNET']['DEPTH_EN'],
                               num_heads=config['SWINUNET']['HEAD_NUM'],
                               window_size=config['SWINUNET']['WIN_SIZE'],
                               mlp_ratio=config['SWINUNET']['MLP_RATIO'],
                               qkv_bias=config['SWINUNET']['QKV_BIAS'],
                               qk_scale=config['SWINUNET']['QK_SCALE'],
                               drop_rate=config['SWINUNET']['DROP_RATE'],
                               drop_path_rate=config['SWINUNET']['DROP_PATH_RATE'],
                               ape=config['SWINUNET']['APE'],
                               patch_norm=config['SWINUNET']['PATCH_NORM'],
                               use_checkpoint=config['SWINUNET']['USE_CHECKPOINTS'])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        original_sizes = (x.shape[2], x.shape[3])
        x = self.process_image_based_on_size(x, original_sizes, (self.config['SWINUNET']['IMG_SIZE'], self.config['SWINUNET']['IMG_SIZE']))
        output = self.swin_unet(x)

        output = self.process_image_based_on_size(output, (self.config['SWINUNET']['IMG_SIZE'], self.config['SWINUNET']['IMG_SIZE']), original_sizes)
        
        return output
    
    def crop_image(self, image, target_height, target_width):
        """ Crop the center of the image to the target size. """
        _, _, current_height, current_width = image.shape

        # Calculate cropping start points
        start_y = (current_height - target_height) // 2
        start_x = (current_width - target_width) // 2

        # Crop the image
        cropped_image = image[:, :, start_y:start_y + target_height, start_x:start_x + target_width]
        return cropped_image
    
    def resize_image(self, image, target_height, target_width):
        """ Resize the image to the target size. """
        resized_image = F.interpolate(image, size=(target_height, target_width), mode='bilinear', align_corners=False)
        return resized_image
    
    def handle_smaller_original(self, image, model_input_size):
        # Resize and then pad
        return self.resize_image(image, *model_input_size)
    
    def handle_larger_original(self, image, original_size, model_input_size):
        # Crop and then resize
        cropped_image = self.crop_image(image, *model_input_size)
        return cropped_image

    def handle_mixed_original(self, image, original_size, model_input_size):
        # Resize first, then crop
        if original_size[0] > model_input_size[0]:
            resized_image = self.crop_image(image, model_input_size[0], original_size[1])
        else: 
            resized_image = self.resize_image(image, model_input_size[0], original_size[1])
        if original_size[1] > model_input_size[1]:
            resized_image = self.crop_image(image, original_size[0], model_input_size[1])
        else: 
            resized_image = self.resize_image(image, original_size[0], model_input_size[1])
        return self.crop_image(resized_image, *model_input_size)

    def process_image_based_on_size(self, image, original_size, model_input_size):
        orig_height, orig_width = original_size
        model_input_height, model_input_width = model_input_size

        if orig_height <= model_input_height and orig_width <= model_input_width:
            # Case 1: Original size is smaller
            return self.handle_smaller_original(image, model_input_size)
        elif orig_height >= model_input_height and orig_width >= model_input_width:
            # Case 2: Original size is larger
            return self.handle_larger_original(image, original_size, model_input_size)
        else:
            # Case 3: Mixed size
            return self.handle_mixed_original(image, original_size, model_input_size)


if __name__ == '__main__':
    import torch
    import yaml
    from thop import profile

    ## Load yaml configuration file
    with open('NN_comp.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    Train = opt['TRAINING']
    OPT = opt['OPTIM']

    height = 822
    width = 1237
    
    x = torch.randn((1, 3, height, width)).cuda()  # .cuda()

    model = SUNet_model(opt).cuda()  # .cuda()
    out = model(x)
    loss = ((out - x) ** 2).mean()
    loss.backward()
    flops, params = profile(model, (x,))
    print(out.size())
    print(flops)
    print(params)