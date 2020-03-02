import os
import subprocess

DIR = os.path.dirname(__file__)

class Synthesiser():
  def __init__(self, blender, target_size=100):
    # Compile configuration settings into a single script file
    self._compile(blender, target_size)
    # Run Blender with the script
    subprocess.call(f'{blender} -t 1 -P script.py -b -noaudio | grep \'^\[synth_motion\]\'', shell=True)

  def _compile(self, blender, target_size):
    configuration = dict({'MOTION_PATH': blender, 'TARGET_SIZE': target_size})

    script_source = open(os.path.join(DIR, 'script-src.py'), 'r').read()
    script_configured = os.path.join(DIR, 'script.py')

    with open(script_configured, 'w') as file:
      for variable, value in configuration.items():
        if type(value) is str:
          file.write(f'{variable} = \'{value}\'\n')
        else:
          file.write(f'{variable} = {value}\n')
      
      file.write(script_source)

