import os

class Synthesiser():
  def __init__(self, blender):

    # TODO: compile
    os.system(f'{blender} -t 1 -P blender-script.py -b | grep \'^\[synth_motion\]\'')
