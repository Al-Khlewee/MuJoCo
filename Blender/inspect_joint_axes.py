import mujoco
from pathlib import Path

model_path = Path(r'c:\users\hatem\Desktop\MuJoCo\mujoco_menagerie\google_barkour_vb\barkour_vb_mjx.xml')
model = mujoco.MjModel.from_xml_path(str(model_path))

print(f"Total joints: {model.njnt}")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    axis = model.jnt_axis[i]
    parent_body = model.jnt_bodyid[i]
    parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_body)
    qposadr = model.jnt_qposadr[i]
    print(f"{i:02d} {name:25s} axis={axis} parent={parent_name} qposadr={qposadr}")
