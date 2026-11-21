EXPERIMENT_NAME=$1
START_NODE=$2
END_NODE=$3
CLUSTER=$4
CLOUDLAB_USERNAME=$5
CLOUDLAB_PROJECTNAME=$6

if [[ $# -lt 6 ]]; then
  echo "Usage (option 1): $0 <experiment_name> <start_node> <end_node> <CLOUDLAB_CLUSTER> <CLOUDLAB_USERNAME> <CLOUDLAB_PROJECTNAME>"
  exit 1
fi

declare -A clusternames
clusternames[emu]="emulab.net"
clusternames[wisc]="wisc.cloudlab.us"
clusternames[utah]="utah.cloudlab.us"

if [[ ! -v clusternames[$CLUSTER] ]]; then
  echo "'$CLUSTER' is not a valid cluster."
  # You can exit or continue as needed
  exit 0
fi

HOSTS=`./cloudlab/nodes.sh $EXPERIMENT_NAME $START_NODE $END_NODE ${clusternames[$CLUSTER]} $CLOUDLAB_USERNAME $CLOUDLAB_PROJECTNAME --all | tr -d ' ' | xargs`
echo "Hosts: $HOSTS"
all_ips=""
for host in $HOSTS; do
    hostname_only=`echo $host | cut -d '@' -f2`
    this_ip=`ping -c 1 $hostname_only | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' | sort -u`
    
    if [[ -z "$this_ip" || $(echo "$this_ip" | wc -l) -gt 1 ]]; then
      echo "Was unable to procure IP address for $host via ping. Trying SSH + ifconfig."
      this_ip=`ssh -o StrictHostKeyChecking=no $host "ifconfig | grep -C4 eno1 | grep -v inet6 | grep inet | tr -s ' ' | cut -d ' ' -f3"`  
      if [[ -z "$this_ip" ]]; then
        echo "Fatal error. SSH+ifconfig also failed to fetch IP"  
        exit 1
      fi
      exit 1
    fi
    all_ips="$all_ips $this_ip"
done
echo "IPs: $all_ips"
