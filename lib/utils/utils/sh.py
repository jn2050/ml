import subprocess

def sh_run(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = proc.communicate()
    return (proc.returncode, stdout.decode("utf-8") , stderr.decode("utf-8") )