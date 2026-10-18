# to be run on rank0 of the training job 

mkdir -p ~/results/stdout/actor_logs
pushd ~/ConstrainedOrca/rl-module/train_dir/
	tar czvf learner_ckpt.tar.gz *
	mv learner_ckpt.tar.gz ~/results/
popd

HOSTS=`python3 -c "import json; import os; f = open(os.path.join('$HOME', 'ConstrainedOrca/rl-module/params_distributed.json')); data = json.load(f)['actor_ip']; f.close(); node_ips=set(map(lambda x: x.split(':')[0], data)); print(' '.join(node_ips))"`
echo "IPs: $HOSTS"

for host in $HOSTS; do
	scp $USER@$host:~/actor_logs/* ~/results/stdout/actor_logs/
done

pushd ~/results
	tar czvf stdout.tar.gz ./stdout/
	rm -rfv ~/results/stdout/
popd

pushd ~/
    tar czvf intermediate_ckpts.tar.gz intermediate_checkpoints/
    mv intermediate_ckpts.tar.gz ~/results/
popd

mkdir ~/results/train_log/

i=1
for host in $HOSTS; do
	ssh $host "cd ~/ConstrainedOrca/rl-module/; tar czvf train_log_node$i.tar.gz training_log/; " &
    i=$((i + 1))
done
wait

i=1
for host in $HOSTS; do
	scp $host:~/ConstrainedOrca/rl-module/training_log/seed0/* ~/results/train_log/
	i=$((i + 1))
done