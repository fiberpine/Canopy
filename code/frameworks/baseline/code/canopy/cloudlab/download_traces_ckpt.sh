# download traces onto current node.
mkdir ~/checkpoints
mkdir ~/traces

wget -O ~/canopy_artefact.zip https://zenodo.org/records/17145110/files/canopy_artefact.zip 
cd ~/ && unzip canopy_artefact.zip
mv canopy_artifact/models/* checkpoints/
mv canopy_artifact/traces/* ~/traces/
rm -rfv canopy_artifact canopy_artefact.zip

git clone https://github.com/Soheil-ab/sage.git
mv sage/traces/ sage/sage_traces/
mv sage/sage_traces/ traces/

ACTOR_NODE_IPS=`python3 -c "import json; f = open('./ConstrainedOrca/rl-module/params_distributed.json'); data = json.load(f)['actor_ip']; f.close(); node_ips=set(map(lambda x: x.split(':')[0], data)); print(' '.join(node_ips))"`

for actor_ip in $ACTOR_NODE_IPS
do  
    scp -r -o StrictHostKeyChecking=no ~/checkpoints/ $USER@$actor_ip:~/
    scp -r -o StrictHostKeyChecking=no ~/traces/ $USER@$actor_ip:~/
done