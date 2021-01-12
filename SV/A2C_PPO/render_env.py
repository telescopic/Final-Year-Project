import PIL
from flatland.utils.rendertools import RenderTool
import numpy as np


# Render the environment
# (You would usually reuse the same RenderTool)
def render_env(env):
    env_renderer = RenderTool(env, gl="PGL")
    env_renderer.render_env()

    image = env_renderer.get_image()
    pil_image = PIL.Image.fromarray(image)
    display(pil_image)
