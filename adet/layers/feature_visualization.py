# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 5/14/22 6:41 PM

import cv2
# import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt






def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def draw_feature_map(features,title = 'title', save_dir = 'feature_map',img_name = None):
    i=0
    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            # img_path = f'/home/wwyu/dataset/mmocr_det_data/icdar2015/imgs/test/{img_name}'
            # save_dir = '/home/wwyu/dataset/mmocr_det_data/icdar2015/vis_feat_ic15_1'
            # img_path = f'/home/wwyu/dataset/mmocr_det_data/ctw1500old/imgs/test/{img_name}'
            # save_dir = '/home/wwyu/dataset/mmocr_det_data/ctw1500old/vis_feat_ctw'
            # img_path = f'/home/wwyu/dataset/mmocr_det_data/total_text/imgs/test/{img_name}'
            # save_dir = '/home/wwyu/dataset/mmocr_det_data/total_text/vis_feat_tt' #可视化改3
            img_name = str(img_name) + '.jpg'
            img_path = f'/root/siton-tmp/SRFormer/datasets/ctw150011111/test_images/{img_name}'
            save_dir = '/root/siton-tmp/SRFormer/datasets/ctw150011111/vis_feat_baseline'

            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            img = cv2.imread(img_path)
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            for idx, heatmap in enumerate(heatmaps):
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap) # 将热力图转换为RGB格式
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # # 将热力图应用于原始图像
                # superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
                superimposed_img = heatmap * 0.5 + img*0.3
                # superimposed_img = heatmap
                # plt.imshow(superimposed_img,cmap='gray') # need BGR2RGB
                # plt.imshow(superimposed_img,cmap='jet')
                # plt.imshow(img,cmap='jet')
                # plt.title(title)
                # plt.show()
                save_file_name = os.path.join(save_dir, f'{os.path.basename(img_path).split(".")[0]}_{title}_'+str(idx)+'.png')
                # cv2.imwrite(save_file_name, superimposed_img)  # 将图像保存到硬盘
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(save_file_name, superimposed_img)

    else:
        for featuremap in features: #可视化改1
            img_path = '/root/siton-tmp/SRFormer/datasets/ctw150011111/test_images/1002.jpg'
            save_dir = '/root/siton-tmp/SRFormer/datasets/ctw150011111/vis_feat'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            img = cv2.imread(img_path)
            heat_maps=heat_maps.unsqueeze(0)

            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for idx, heatmap in enumerate(heatmaps):
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # 将热力图应用于原始图像
                superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
                # superimposed_img = heatmap * 0.5 + img*0.3
                # superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                # plt.imshow(superimposed_img,cmap='jet')
                plt.title(title)
                plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                cv2.imshow("1",superimposed_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,'vis' +str(i)+'.png'), superimposed_img) #可视化改2
                i=i+1

from .ExplanationGenerator import Generator


def evaluate(model, gen, im, device, image_id = None):
    # mean-std normalize the input image (batch-size: 1)
    img_path = f'/root/siton-tmp/SRFormer/datasets/ctw1500/test_images/1001.jpg'
    img = cv2.imread(img_path)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    if keep.nonzero().shape[0] <= 1:
        return

    outputs['pred_boxes'] = outputs['pred_boxes'].cpu()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        # model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
        #     lambda self, input, output: enc_attn_weights.append(output[1])
        # ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    for layer in model.transformer.encoder.layers:
        hook = layer.self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        )
        hooks.append(hook)

    model(img)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[-1]
    dec_attn_weights = dec_attn_weights[0]

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]
    img_np = np.array(im).astype(np.float)


    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        ax = ax_i[0]
        cam = gen.generate_ours(img, idx, use_lrp=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cmap = plt.cm.get_cmap('Blues').reversed()
        ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap)
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')
        ax = ax_i[1]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin.detach(), ymin.detach()), xmax.detach() - xmin.detach(), ymax.detach() - ymin.detach(),
                                   fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
    id_str = '' if image_id == None else image_id
    fig.tight_layout()
    plt.show()