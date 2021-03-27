from flatland.envs.rail_env import RailEnv
from flatland.utils.misc import str2bool
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv

from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file

from flatland.envs.agent_utils import RailAgentStatus
from flatland.utils.rendertools import RenderTool

import PIL

observation_tree_depth = 2
observation_radius = 10
observation_max_path_depth = 30

predictor = ShortestPathPredictorForRailEnv(observation_max_path_depth)
tree_observation = TreeObsForRailEnv(max_depth=observation_tree_depth, predictor=predictor)

env_file = "D:/Sudhish/FYP/Final-Year-Project-main/Sudhish/envs-100-999/envs/Level_101.pkl"




env = RailEnv(width=1, height=1,
                      rail_generator=rail_from_file(env_file),
                      schedule_generator=schedule_from_file(env_file),
                      malfunction_generator_and_process_data=malfunction_from_file(
                          env_file),
                      obs_builder_object=tree_observation)

obs, info = env.reset(True, True)
env_renderer = RenderTool(env, gl="PILSVG")
env_renderer.render_env()
image = env_renderer.get_image()
pil_image = PIL.Image.fromarray(image)
pil_image.show()
print("Env Loaded")