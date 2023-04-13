import base64
import io
import os
import sys

import numpy as np
# diffusers
import requests
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download
# segment anything
from segment_anything import build_sam, SamPredictor

# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

app = Flask(__name__)
CORS(app, resources=r'/*')  # 注册CORS, "/*" 允许访问所有api

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

# If you have multiple GPUs, you can set the GPU to use here.
# The default is to use the first GPU, which is usually GPU 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))

retry = 5


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


def download_image(url, image_file_path):
    r = requests.get(url, timeout=10)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)

    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))


def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


@app.route('/detect', methods=['POST'])
def do_detect():
    image_union = []

    # 获取上传的文件数据
    file = request.files['file']

    # 获取额外的参数
    param = request.form.get('prompt')

    print(param)

    body = request.form.get('body')

    # 将文件保存到磁盘
    file.save('./uploads/' + file.filename)

    while len(image_union) < 4:
        try:
            groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16,
            )

            pipe = pipe.to("cuda")

            # download_image(image_url, local_image_path)
            local_image_path = './uploads/' + file.filename

            TEXT_PROMPT = body
            BOX_TRESHOLD = 0.3
            TEXT_TRESHOLD = 0.25

            image_source, image = load_image(local_image_path)

            boxes, logits, phrases = predict(
                model=groundingdino_model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )

            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB

            # 原图
            # Image.fromarray(image_source)
            with io.BytesIO() as output:
                Image.fromarray(image_source).save(output, format='PNG')
                image_data = output.getvalue()
            image_union.append(image_data)
            # 识别物体
            # Image.fromarray(annotated_frame)
            with io.BytesIO() as output:
                Image.fromarray(annotated_frame).save(output, format='PNG')
                image_data = output.getvalue()
            image_union.append(image_data)

            # set image
            sam_predictor.set_image(image_source)

            # box: normalized box xywh -> unnormalized xyxy
            H, W, _ = image_source.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2])
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            annotated_frame_with_mask = show_mask(masks[0][0], annotated_frame)

            # 分割后画面
            # Image.fromarray(annotated_frame_with_mask)
            with io.BytesIO() as output:
                Image.fromarray(annotated_frame_with_mask).save(output, format='PNG')
                image_data = output.getvalue()
            image_union.append(image_data)

            image_mask = masks[0][0].cpu().numpy()

            image_source_pil = Image.fromarray(image_source)
            annotated_frame_pil = Image.fromarray(annotated_frame)
            image_mask_pil = Image.fromarray(image_mask)
            annotated_frame_with_mask_pil = Image.fromarray(annotated_frame_with_mask)

            # 切分后画面
            # image_mask_pil
            with io.BytesIO() as output:
                image_mask_pil.save(output, format='PNG')
                image_data = output.getvalue()
            image_union.append(image_data)

            # resize for inpaint
            image_source_for_inpaint = image_source_pil.resize((512, 512))
            image_mask_for_inpaint = image_mask_pil.resize((512, 512))

            prompt = param
            # image and mask_image should be PIL images.
            # The mask structure is white for inpainting and black for keeping as is
            image_inpainting = \
                pipe(prompt=prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]

            image_inpainting = image_inpainting.resize((image_source_pil.size[0], image_source_pil.size[1]))

            # 成品画面
            # image_inpainting
            with io.BytesIO() as output:
                image_inpainting.save(output, format='PNG')
                image_data = output.getvalue()
            image_union.append(image_data)

            for i in range(len(image_union)):
                image_data = image_union[i]
                encoded_data = base64.b64encode(image_data).decode('utf-8')
                image_union[i] = encoded_data

            break  # 如果没有出错，跳出循环
        except:
            print("发生错误，重新执行")
    return jsonify({'images': image_union})


if __name__ == '__main__':
    app.run(debug=True)
