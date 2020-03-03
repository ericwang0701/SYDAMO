import os
import subprocess
import logging

dirname = os.path.dirname(__file__)
source = os.path.join(dirname, 'script-src.py')
compiled = os.path.join(dirname, 'script.py')


class Synthesiser():
    def __init__(self, blender, motion_path, target_size, num_frames, debug_blender):
        logging.basicConfig(level='INFO', format='%(message)s')
        logging.info('=================== SYNTHESISER ===================')

        self.blender = os.path.join(os.getcwd(), blender)
        self.debug_blender = debug_blender

        self.configuration = dict({
            'MOTION_PATH': motion_path,
            'TARGET_SIZE': target_size,
            'MAX_FRAMES': num_frames,
            'NO_RENDER': debug_blender
        })

    def run(self):
        # Compile configuration settings into a single script file
        self._compile()
        # Run Blender with the script
        blender_cmd = [self.blender, '-t', '1', '-P', compiled, '-noaudio']

        # Do not open Blender GUI unless debug is turned on
        if not self.debug_blender:
            blender_cmd.append('-b')

        subprocess.run(blender_cmd)

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
