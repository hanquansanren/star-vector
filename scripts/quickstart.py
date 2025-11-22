import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from PIL import Image
from starvector.model.starvector_arch import StarVectorForCausalLM
from starvector.data.util import process_and_rasterize_svg
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_name = os.path.join(
    "/Data_PHD/phd23_weiguang_zhang/project/weights",
    "models--starvector--starvector-1b-im2svg/snapshots/380ab95d25a8e9ab1dc825debe238b4953ae13b9"
)
# model_name = "starvector/starvector-1b-im2svg"
# model_name = "starvector/starvector-8b-im2svg"

starvector = StarVectorForCausalLM.from_pretrained(model_name, torch_dtype="auto", local_files_only=True) # add , torch_dtype="bfloat16"

starvector.cuda()
starvector.eval()


file_name = "hei3.png"
image_pil = Image.open(os.path.join(project_root, "assets", "temp", file_name))
image_pil = image_pil.convert('RGB')
image = starvector.process_images([image_pil])[0].to(torch.float16).cuda()
batch = {"image": image}

raw_svg = starvector.generate_im2svg(batch, max_length=4000, temperature=1.5, length_penalty=-1, repetition_penalty=3.1)[0]
svg, raster_image = process_and_rasterize_svg(raw_svg)


# 确保输出目录存在
output_dir = os.path.join(project_root, "outputs")
os.makedirs(output_dir, exist_ok=True)

# 保存SVG文件
output_svg_path = os.path.join(output_dir, "output.svg")
with open(output_svg_path, "w", encoding="utf-8") as f:
    f.write(svg)
print(f"\n✓ IMG2SVG: {os.path.abspath(file_name)}")
print(f"\n✓ SVG已保存到: {os.path.abspath(output_svg_path)}")

# 保存渲染后的图像
output_image_path = os.path.join(output_dir, "output_rasterized.png")
raster_image.save(output_image_path)
print(f"✓ 渲染后的图像已保存到: {os.path.abspath(output_image_path)}")

# 打印SVG内容的前500个字符供预览
print(f"\n生成的SVG预览（前500字符）:")
print(svg[:500] + "..." if len(svg) > 500 else svg)