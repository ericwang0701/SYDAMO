import os

class Synthesiser():
  def __init__(self):
    # TODO: compile
    os.system('blender -t 1 -P blender-script.py | grep \'^\[synth_motion\]\'')
