import os
import subprocess

dirname = os.path.dirname(__file__)
source = os.path.join(dirname, 'script-src.py')
compiled = os.path.join(dirname, 'script.py')

class Synthesiser():
  def __init__(self, blender, motion_path, target_size):
    self.blender = blender

    self.configuration = dict({
      'MOTION_PATH': motion_path,
      'TARGET_SIZE': target_size
    })

  def run(self):
    # Compile configuration settings into a single script file
    self._compile()
    # Run Blender with the script
    subprocess.call(f'{self.blender} -t 1 -P {compiled} -b -noaudio | grep \'^\[synth_motion\]\'', shell=True)

  def _compile(self):
    script_source = open(source, 'r').read()
    
    with open(compiled, 'w') as file:
      for variable, value in self.configuration.items():
        if type(value) is str:
          file.write(f'{variable} = \'{value}\'\n')
        else:
          file.write(f'{variable} = {value}\n')
      
      file.write(script_source)
