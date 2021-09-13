srun -N 1 \
     --container-image=nvcr.io/nvidia/kaldi:21.08-py3 \
     --container-mounts /gpfs/fs1/dgalvez/code/kaldi/:/kaldi-work-dir/,/gpfs/fs1/datasets/people-speech:/gpfs/fs1/datasets/people-speech \
     --time=01:00:00 \
     --pty \
     /bin/bash -i

#          --container-image=gitlab-master.nvidia.com/dgalvez/nemo:latest \
