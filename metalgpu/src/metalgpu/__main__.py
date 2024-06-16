import os
import subprocess

curr_path = os.path.dirname(__file__)


def build():
    mkdir_path = f"mkdir -p {curr_path}/lib"
    cd_to_path = f"cd {curr_path}/lib"
    clone_git_repo = "git clone https://github.com/Al0den/metalgpu"
    cd_to_c_code = "cd metalgpu/metal-gpu-c"
    clone_metal_cpp = "git clone https://github.com/Al0den/metal-cpp-extension"
    rename_metal_cpp = "mv metal-cpp-extension metal-cpp"
    cmake_cmd = "cmake ."
    install = "make install"
    copy_to_correct_path = "cp ./libmetalgpucpp-arm.dylib ../.."
    remove_prev_folder = "rm -rf metalgpu"
    remove_down_folder = "rm -rf ../../metalgpu"

    final_command = f"{mkdir_path} && {cd_to_path} && {remove_prev_folder} && {clone_git_repo} && {cd_to_c_code} && {clone_metal_cpp} && {rename_metal_cpp} && {cmake_cmd} && {install} && {copy_to_correct_path} && {remove_down_folder}"
    subprocess.run(final_command, shell=True)


build()
