"""
Step 2. Get reward feedback for generated images in the output folder
"""

from gid_tools.envs.feedback import ToolRewardEnv, pixel_area_tensor



# start with an empty registry
env = ToolRewardEnv(default_method=None)       
env.register_reward('pixel_area', pixel_area_tensor)


# this gets reward feedback on all generated images

# go to samples/ sample_i.png

# load in gid_tools.envs.feedback

# iterate through samples folder and get reward feedback

# save 