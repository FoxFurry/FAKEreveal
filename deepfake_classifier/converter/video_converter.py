import subprocess

def web_to_mp4(input_file, output_file):
    try:
        input_file = r'{}'.format(input_file)
        output_file = r'{}'.format(output_file)
        command = 'ffmpeg -i ' + input_file + ' -r 30 ' + output_file
        subprocess.run(command)
        return 1
    except Exception as e:
        return 0