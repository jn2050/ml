import subprocess
import zipfile
import os


def sh_run(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdout, stderr) = proc.communicate()
    return (proc.returncode, stdout.decode("utf-8") , stderr.decode("utf-8") )


def zipdir(dir, fname):
    zf = zipfile.ZipFile(fname, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(dir):
        for file in files:
            zf.write(os.path.join(root, file))