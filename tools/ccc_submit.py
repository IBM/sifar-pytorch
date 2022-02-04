#!/usr/bin/env python3

import subprocess
import argparse
import os
import platform

parser = argparse.ArgumentParser(description='BSUB job submission')
parser.add_argument('-j', '--job_name', default='', help='job name and log name prefix (Default: my_job)')
parser.add_argument('-n', '--nodes', default=1, type=int, help='number of nodes (Default: 1)')
parser.add_argument('-ng', '--num_gpus', default=1, type=int, help='number of gpus per node (Default: 1)')
parser.add_argument('-nc', '--num_cores', default=0, type=int, help='number fo cores per node, if 0, will use the value: num_gpus * 6')
parser.add_argument('--mem', default=0, type=int, help='cpu memory per node (GB), if 0, will use the value: num_gpus * 32G')
parser.add_argument('--dep', default=None, type=str)
parser.add_argument('-q', '--queue', default='x86_24h', type=str, help='queue, check via bqueue')
parser.add_argument('-nd', '--num_deps', default=1, type=int, help='number of dependent jobs, including the submitted one (Default: 0)')
parser.add_argument('--model', default='', help='model name')
parser.add_argument('--suffix', default='', help='suffix name append to model as log folder name')
parser.add_argument('--job_dir', default="checkpoint/experiments/", help='log path')
parser.add_argument('cmd', help='whole command, quoted by ""', metavar='CMD')

args = parser.parse_args()


def main():

    log_folder = args.model if args.suffix == '' else args.model + "-" + args.suffix
    log_folder = os.path.join(os.getcwd(), args.job_dir, log_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    job_name = os.path.basename(log_folder) if args.job_name == '' else args.job_name

    args.cmd = args.cmd + f" --output_dir {log_folder} --model {args.model}"

    num_cores = args.num_gpus * 6 if args.num_cores == 0 else args.num_cores
    mem = args.num_gpus * 32 if args.mem == 0 else args.mem
    mem = str(mem) + 'G'

    hw_cfg = f'{args.nodes}x{num_cores}+{args.num_gpus}'
    
    for i in range(args.num_deps):
        cmd = args.cmd
        if i == 0:
            if args.dep:
                dep_option = f' -depend "ended({args.dep})" '
            else:
                dep_option = ' '
        else:
            dep_option = f' -depend "ended({job_id})" '
            cmd = cmd + f" --resume {log_folder}/checkpoint.pth"
            
        job=f"""#!/bin/bash
#---------------------------------------
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_CUDA_SUPPORT=1
export NGPUS={args.num_gpus}
export NODES={args.nodes}
jbsub -name {job_name} -cores {hw_cfg} -mem {mem} -q {args.queue} -e {log_folder}/%J.err -o {log_folder}/%J.out {dep_option} -require v100 blaunch.sh ./tools/train.sh main.py {cmd}
#---------------------------------------
"""
        print(job)
        script = "{}/{}.sh".format(log_folder, job_name)
        print("Generate script at {}".format(script))
        with open(script, 'w') as f:
            print(job, file=f, flush=True)

        p = subprocess.Popen(['sh', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        stdout = stdout.decode("utf-8")
        job_id = stdout.split("<")[1].split(">")[0].strip()
        print(f"Job {job_id.strip()} is submitted.")
        
        
if __name__ == "__main__":
    main()
