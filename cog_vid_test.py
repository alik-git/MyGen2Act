import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

prompt = "A robot arm moves the can in the middle bottom of the table"
image = load_image(image="/home/kasm-user/alik_local_data/first_episode_images/Move_the_can_in_the_middle_bottom_of_the_table.png")
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b-I2V",
    torch_dtype=torch.bfloat16
)

pipe.enable_sequential_cpu_offload()
# pipe.vae.enable_tiling()
# pipe.vae.enable_slicing()

video = pipe(
    prompt=prompt,
    image=image,
    num_videos_per_prompt=1,
    num_inference_steps=3,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "/home/kasm-user/Downloads/output2.mp4", fps=8)