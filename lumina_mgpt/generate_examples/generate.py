import os
import sys
sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])
import argparse
from PIL import Image
import torch
from inference_solver import FlexARInferenceSolver
sys.path.append(os.path.abspath(__file__).rsplit("/", 3)[0])
from xllmx.util.misc import random_seed
import time
from jacobi_utils_static import renew_pipeline_sampler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--task", type=str, default='t2i')
    parser.add_argument("--speculative_jacobi", default=False, action='store_true')
    parser.add_argument("--quant", default=False, action='store_true')

    args = parser.parse_args()

    print("args:\n", args)
    l_prompts = [
        "Image of a dog playing water, and a water fall is in the background.",
        "A high-resolution photograph of a middle-aged woman with curly hair, wearing traditional Japanese kimono, smiling gently under a cherry blossom tree in full bloom.",  # noqa
        "A pink and chocolate cake ln front of a white background",
        "A highly detailed, 3D-rendered, stylized representation of 2 travellers, a 40 year old man and a step behind, a 40 year old woman walking on a path. Full body visible. They have large, expressive hazel eyes and dark, curly hair that is slightly messy. Their faces are full of innocent wonder, with rosy cheeks and a scattering of light freckles across his nose and cheeks. They are wearing thicker clothes, trousers and hiking shoes, making it look cosy. The background is softly blurred with warm tones, putting full focus on their facial features and expressions. The image has a soft, cinematic lighting style with subtle shadows that highlight the contours of their faces, giving it a realistic yet animated look. The overall art style is similar to modern animated films with high levels of detail and a slight painterly touch, 8k.",
        "Indoor portrait of a young woman with light blonde hair sitting in front of a large window. She is positioned slightly to the right, wearing an oversized white shirt with rolled-up sleeves and brown pants. A black shoulder bag is slung over her left shoulder. Her lips are pursed in a playful expression. The window behind her features closed horizontal blinds, reflecting faint interior lighting. The surrounding wall is made of textured, light-colored brick. The lighting is soft, highlighting her features and the textures of her clothing and the wall. Casual, candid, slightly cool color temperature, natural pose, balanced composition, urban interior environment."
    ]

    t = args.temperature
    top_k = args.top_k
    cfg = args.cfg
    n = args.n
    w, h = args.width, args.height
    device = torch.device("cuda")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    inference_solver = FlexARInferenceSolver(
        model_path=args.model_path,
        precision="bf16",
        quant=args.quant,
        sjd=args.speculative_jacobi,
    )
    print("checkpiont load finished")

    if args.speculative_jacobi:
        print(inference_solver.__class__)
        print("Use Speculative Jacobi Decoding to accelerate!")
        max_num_new_tokens = 16
        multi_token_init_scheme = 'random' # 'repeat_horizon'
        inference_solver = renew_pipeline_sampler(
            inference_solver,
            jacobi_loop_interval_l = 3,
            jacobi_loop_interval_r = (h // 8)**2 + h // 8 - 10,
            max_num_new_tokens = max_num_new_tokens,
            guidance_scale = cfg,
            seed = None,
            multi_token_init_scheme = multi_token_init_scheme,
            do_cfg=True,
            image_top_k=top_k, 
            text_top_k=10,
            prefix_token_sampler_scheme='speculative_jacobi',
            is_compile=args.quant
        )

    with torch.no_grad():
        for i, prompt in enumerate(l_prompts):
            for repeat_idx in range(n):
                random_seed(repeat_idx)
                if args.task == 't2i':
                    generated = inference_solver.generate(
                            images=[],
                            qas=[[f"Generate an image of {w}x{h} according to the following prompt:\n{prompt}", None]],  # high-quality synthetic  superior
                            max_gen_len=10240,
                            temperature=t,
                            logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k),
                        )
                else:
                    task_dict = {"depth": "depth map", "canny": "canny edge map", "hed": "hed edge map", "openpose":"pose estimation map"}
                    generated = inference_solver.generate(
                            images=[],
                            qas=[[f"Generate a dual-panel image of {w}x{h} where the <lower half> displays a <{task_dict[args.task]}>, while the <upper half> retains the original image for direct visual comparison:\n{prompt}" , None]], 
                            max_gen_len=10240,
                            temperature=t,
                            logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k),
                            )
                generated[1][0].save(args.save_path + f"{i}.png")
