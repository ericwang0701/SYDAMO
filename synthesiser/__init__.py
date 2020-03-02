import os
import subprocess
import logging

dirname = os.path.dirname(__file__)
source = os.path.join(dirname, 'script-src.py')
compiled = os.path.join(dirname, 'script.py')

class Synthesiser():
  def __init__(self, blender, motion_path, target_size, num_frames):
    logging.basicConfig(level='INFO', format='%(message)s')
    logging.info('=================== SYNTHESISER ===================')

    self.blender = os.path.join(os.getcwd(), blender)

    self.configuration = dict({
      'MOTION_PATH': motion_path,
      'TARGET_SIZE': target_size,
      'MAX_FRAMES': num_frames
    })

  def run(self):
    # Compile configuration settings into a single script file
    self._compile()
    # Run Blender with the script
    subprocess.call(f'{self.blender} -t 1 -P {compiled} -b -noaudio | grep \'^\[synthesiser\]\'', shell=True)

  def _compile(self):
    script_source = open(source, 'r').read()
    
    with open(compiled, 'w') as file:
      for variable, value in self.configuration.items():
        if type(value) is str:
          file.write(f'{variable} = \'{value}\'\n')
        else:
          file.write(f'{variable} = {value}\n')
      
      file.write(script_source)
    
    assert(os.path.exists(compiled))
