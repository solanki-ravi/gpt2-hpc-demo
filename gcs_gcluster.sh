project_id=ambient-cubist-384700
git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git
cd cluster-toolkit/
make
./gcluster --version
./gcluster create examples/hpc-slurm.yaml \
    -l ERROR --vars project_id=$project_id
./gcluster deploy hpc-slurm
./gcluster status

# ssh to the head node
gcloud compute ssh head-0 --zone=us-central1-a

# run a command on the head node
srun -N 3 hostname

# destroy the cluster
./gcluster destroy hpc-slurm-gpt2demo --auto-approve
