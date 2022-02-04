#!/usr/bin/env python3

import subprocess
import argparse
import os
import platform

parser = argparse.ArgumentParser(description='BSUB job submission')
parser.add_argument('-j', '--job_name', default='', help='job name and log name prefix (Default: my_job)')
parser.add_argument('-t', '--time', default=72, type=int, help='hours (Default: 72 hours)')
parser.add_argument('--wd', default=os.getcwd(), type=str, help='root of working directory (Default: ./)')
parser.add_argument('-n', '--nodes', default=1, type=int, help='number of nodes (Default: 1)')
parser.add_argument('--dep', default=None, type=str)
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

    if args.dep:
        dep_option = f'#BSUB -w ended({args.dep})'
    else:
        dep_option = ''


    job=f"""#!/bin/bash

cat > {log_folder}/batch.job <<EOF
#BSUB -J {job_name}
#BSUB -o {log_folder}/%J.out
#BSUB -e {log_folder}/%J.err
#BSUB -nnodes {args.nodes}
#BSUB -q excl
#BSUB -W {args.time}:00
{dep_option}

#---------------------------------------
ulimit -s unlimited
ulimit -c 100000

export OMP_NUM_THREADS=1
cd {args.wd}
export MASTER_HOSTNAME=\$(cat \$LSB_DJOB_RANKFILE | tail -n +2 | sort -V | head -n 1)
export NODES={args.nodes}
jsrun --bind none -E WSC=1 -n {args.nodes} -r 1 -D CUDA_VISIBLE_DEVICES -a 1 -c ALL_CPUS -g ALL_GPUS tools/train.sh main.py {args.cmd}
EOF
#---------------------------------------
bsub  < {log_folder}/batch.job

"""
    #awk \"{{ print \$0 \\" slots=1\\"; }}\" ~/tmp/hosts.\$LSB_BATCH_JID.tmp > ~/tmp/tmp.\$LSB_BATCH_JID
    print(job)
    script = "{}/{}.sh".format(log_folder, job_name)
    print("Generate sbatch script at {}".format(script))
    with open(script, 'w') as f:
        print(job, file=f, flush=True)

    os.system(f'sh {script}')


if __name__ == "__main__":
    main()
