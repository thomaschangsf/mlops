# Setup
- githubs
	- [github-advk8s - thomaschangsf](https://github.com/thomaschangsf/k8s_adv)
	- [github-dockerK8s - thomaschangsf](https://github.com/thomaschangsf/k8s_material)
- instructor email: jkidd@kiddcorp.com
- local mac:
	- /Users/thomaschang/git/thomaschangsf/k8s_adv
- References
	- Main class notes are in the githubs above
		- local copies are also in tools/Rerferences/DockerK8sHelm
		- Class went over Kubernetes_Adv_1&2.pdf
	




# 1  Replicasets
- purpose: maintain redundant services resistant to failure


# 2 Deployments
- purpose manage rollouts of application containers without downtime while maintaining redundancy
- IMOW
	- deployment
		-  pods * replicaset (scale horizontally via replicatset)
			- 1 pod = C containers
			- scale pod horizontally via HPA

# 3 Probes
- purpose Gives k8s insights into your service's health

- 2 types: readiness vs liveness
	- readiness: when service can be routed to this pod
	- liveness: is the pod still alive?
	- kubelett can only see if pod is running, 
		- but  pod can still be unusable; 
			- ex: has a dead loack, kubelete will not see this.
	- comparisoin
		- one can usually use a liveness as a readiness has a delay before request can be sent
		- but liveness has a failurecount, exceeding will cause a restart

- Health checks ways used by kubelete (pg k8 in kubernetes_2.pdf)
	- http (status 200-399)
	- container exec; exit status = 0
		- ie exec: command: cat /tmp/
	- tcp socket

- LAB1
	- Directory: samples/probe.yaml
	- ktl apply -f probe.yaml
		- image will fail intermittently; kubelete will restart the liveness probe
	- ktl describe pods/liveness-http

- LAB2
	- probe-2.yaml
		- uses liveness probe
	- 

# 4 PodDisruptionBudgets
- purpose prevent application disruptions
- Relates to scheduler (see section 9) below
	- Scheduler labuses scheduler actions(drain) to trigger PDB violations


# 5a Managing Container Resources
- There are two ways Kubernetes controls resources such as CPU and memory: 
	• Requests – This is the value the container is guaranteed to get when it’s pod is scheduled. If the scheduler can’t find a node with this amount, then the pod wont get scheduled. 
	• Limits – This it the limit placed on the CPU or memory. The container will never use more than this.
- These values are assigned to ==containers==, not pods. The resourc efr a pod is the sum of all the containers. 
	- k8s pod vs container
		- Kubernetes pods are collections of containers that share the same resources and local network
		- This enables easy communication between containers in a pod.
		- More details: pods vs container vs nodes
			- Containers (lowest) are packages of applications and execution environments.
			- Pods are collections of closely-related or tightly coupled containers.
			- Nodes (higher) are computing resources that house pods to execute workloads.
	- Units
		- CPU : express in xxxm format
			- 0.1 =. 100m  = 100 millicores
			- ==1 core = 1000 millicore==
		- Memory: xxMi, xxGi
			- M = MegaBytes
	- Example: 1 pod = N containers, where a container can be a initContainer or sidecar
		- total for pod: CPU request = 300
		```yaml
		containers: 
			- name: container1 
				- image: myimage:v1 
				- resources: 
					- requests: 
						- memory: “64Mi” 
						- cpu: ”200m” 
					- limits: 
						- memory: “128Mi” 
						- cpu: “600m” 
			- name: container2 image: myotherimage:v1 resources: requests: memory: “32Mi” cpu: ”100m” limits: memory: “64Mi” cpu: “300m” 
		```
			
- Request vs Limits
	- Requests – This is the value the container is guaranteed to get when it’s pod is scheduled. If the scheduler can’t find a node with this amount, then the pod wont get scheduled. 
	- Limits – This it the limit placed on the CPU or memory. The container will never use more than this.
	- Request can never be higher than limits
	- 
- ResourceQuota limit resources at the namespace level
	- Earlier, we set at pod level container or namespace level



# 5b Resource Metric Collection
- Purpose: to enable monitoring and decision making
	- Ex: Autoscaling pods and cluster
- Metric servier tool collectsion mutliple metrics
	- typical: cpu and memory
	- other: diskio, request count
- Alternative to MetricServer
	- Prometheus
		- plugs in well with graphana for monitoring


# 5c: Lab
- LAB
	- php-apache.yaml
		- requires:
			- metrics: we will use metric server solution (there are others)
				- components.yaml in the repo sets this metric-server
					- --kubelet-insecure-tls : this is for testing, not secure
					- k8s apply -f component.yaml
						- creat rbac (role based authentication)
						- ktl get all -n kubesystem
		- ktl apply -f php-apache.yaml
			- ktl top pods
		- Autoscale
			- kubectl autoscale deployment php-apache --cpu-percent=50 --min=1 --max=10
		- Generate load
			- kubectl run -i --tty load-generator --rm --image=busybox:1.28 --restart=Never -- /bin/sh -c "while sleep 0.01; do wget -q -O- http://php-apache; done"
		- Watch from another window
			- kubectl get hpa php-apache --watch
			- if want more load, use another terminal  with a different name for the auto=scaler


# 6 Horizontal Pod AutoScaler
- Purpose: dynamically size replicaset in response to demand
- Overview
	- Scaling pods (horizontally or vertically)
		- horizontally: add/remove ==copies== of resources
		- vertical: make unit of work bigger or smaller
	- Scaling clusters
- See lab in section 5c above
- HPA is based on aggregate metric of all pods
	- if there is a spike in load for 1 pod, use prometheus to look.
- If there is a conflict in replicaset between the deployment.yaml and hpa, hpa's value will take precedence



# 7 Cluster Autoscaler
- purpose: resize entire cluster



# 8 Initcontainers
- purpose: dependencies, configuration, and initialization
	- ==Init containers are essentially an idiom for expressing dependency== 
		- running to completion can mean different things
		- Ex usage:
			- dependency checks
			- file copying
- Init containers is very important, helps us when pod can receive request
	- in addition to readiness probe
	- init continer is to check whether the resources are ready (ie db, container iages downloaded)
	- Ex: p159 of kubernetes_2.pdf
- Examples:
	- sample/probe-2.yaml
	- Advanced Kubernetes/initContainer/init.yaml
		``` yaml
		containers:  
			- name: myapp-container  
			  image: busybox:1.28  
			  command: ['sh', '-c', 'echo The app is running! && sleep 3600']  
			initContainers:  
			- name: init-myservice  
			  image: busybox:1.28  
			  command: ['sh', '-c', 'until nslookup myservice; do echo waiting for myservice; sleep 2; done;']  
			- name: init-mydb  
			  image: busybox:1.28  
			  command: ['sh', '-c', 'until nslookup mydb; do echo waiting for mydb; sleep 2; done;']
		```
		
		- ktl create ns advK8s 
		- ktl apply -f service.yaml
			- to create mydb and myservice services
		- ktl apply -f init.yaml
		- ktl describe myapp-pod -n advk8s

- Docker compose 
	- is useful to manage dependencies between components
		- more powerful than initContainers
	- example: [acdc](https://github.com/thomaschangsf/acdc)
		```yaml 
		version: "3.8"
		
		services:
		  ws:
		    image: realstraw/sbt
		    working_dir: /acdc/ws
		    volumes:
		      - ./:/acdc/ws
		    environment:
		      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}"
		    ports:
		      - 9001:9000
		    tty: true # for easy debugging
		    # command:
		    #   sbt ws/run
		  db:
		    image: postgres
		    volumes:
		      - acdc-postgres-data:/var/lib/postgresql/data
		    environment:
		      POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}"
		    ports:
		      - 5432:5432
		  flyway:
		    image: flyway/flyway:latest-alpine
		    volumes:
		      - ./flyway/sql:/flyway/sql
		    environment:
		      FLYWAY_URL: "jdbc:postgresql://db:5432/postgres"
		      FLYWAY_USER: "postgres"
		      FLYWAY_PASSWORD: "${POSTGRES_PASSWORD}"
		    depends_on:
		      - db
		    command:
		      migrate
		
		volumes:
		  acdc-postgres-data:
```

# 9 Schedule
- pg 120 in kubernetes_1.pdf
- purpose: How cluster decide which pod to deploy in which cluster
	- Cluster = N Nodes
		- 6 GPUS
		- 15 CPUS
- Uses by default
	- CPU and memory consumptin
	- your deployment's request resources
- Other scheduling pacements (p116 of Kubernetes_1.pdf in git)
	-  to prevent a pod to be deployed in gpu node, use 
		- ==node taints==
			- ==toleration== enables one to overcome a node -tant
		- node labels
			- labels tell pods which nodes thay should be scheduled on 
			- builtin labels
				- hostname, zone, region
				- instance-type
		- node affiniity
			- control to which pod is deployed in which machine
			- simmilar to nodel labels but more epxressive
				- use expressions: in, notin, lt, gt
				- 
		- Summary
			- Node labels —> White listing, Taint —> Blacklisting
		- Salesforce
			- Uses (Open Policy Agent) OPA that allows manage these kind of questions.
- LAB (pod descrition budget - see section 2)
	- ==This lab uses scheduler actions(drain) to trigger PDB violations==
		 
	- PDB proectes againsts ==volunatary== actions
		- no protection: delete, drain
		- voluntary: drain
	- zk.yaml (zookeeper)
		- zookeeper is a quorum based tool to manage distributed serivces
			- requires more than 1 instance to be running
		- stateful set:  limit use them because you are responsible for 
			- stateful set is good for storage
	- ktl apply -f zk.yaml
		- deploy 3 pods across 3 cluster
	- ktl get pdb
		- tells me I can lose 1 pod
	- ktl drain cluster-worker --ignore-daemonsets --delete-emptydir-data
		- evicts the node from this cluster
		- because of the zk. pdb, the drain command fails

- Custom Scheduler (p156)
	- You can create/update your own scheduler
	- Logic
		1 A loop to watch the unbound pods in the cluster through querying the apiserver 
		2 Custom logic that finds the best node for a pod. 
		3 A request to the bind endpoint on the apiserver.
		``` yaml
		apiVersion: v1 
		kind: Pod 
		spec: 
			schedulerName: my-scheduler 
			containers: 
				- name: nginx 
				- image: nginx:1.10
		```
		- can look for open source
	- Lab:
		- ./k8s_adv/labs/03-scheduler/manifests/scheduler.yaml
		- k8s  apply -f Advanced\ Kubernetes/initContainers/init.yaml
			- pending state --> why ???
				- need scripts to be executed
			- ./scripts/scheduler.sh
				- need to match the port 
				- sh can be a go script as well
				




# 11 Networking
- Ref: Kubernetes_2.pdf




# 12 HELM
- Ref: Kubernetes_2.pdf (pg 70)
- Helm 
	- enable dependency managment
		- we may hav sql db, web application
		- it is the package manager to deal with multiple tiers (p76)
			- system
				- apt: deb
				- yum: rpm
			- dev
				- mavn: jar, ear
				- pip: python package
			- k8s 
				- helm :charts
				
	- templatize to support environments

- Use
	- help repo list
		- repository: stable, bitnami, grafana
	- helm add repo [name URL]
	- helm repo update
	- helm search repo mysql
	- helm pull bitnami/sql
		- pull down the archiv show what it is 
			- ==good way to study skill up on helm by reverse engineering==
		- tar xvf mysql-9.7
	- helm install mysql bitnamy/mysql
		- install mysql
	- helm list
		- show 


- Lab
	- helm create my chart
		- create a direcotries: Chart.yaml, charts, templates
		- creates values
			- values.yaml: create variables used in manifest file
		- charts
			- can dependency dependency to other charters
		- templates (valid k8s yaml file)
			- Resources
				- deployment.yaml
				- ingress.yaml
				- serice.yaml
			- All resource can refer to the values.yaml
	- helm 





# Appendix:

### OTHERS
- Use namespaces
	- allows security and resource management at namespace level
	- Wallmart can deploy across all cloud throught the use namespace
	- there is a tool called rancher, to deploy across cluster and clouds
	- 


### K8s Commands
- ktl apply -f php-apache.yaml
- ktl top pods -n kube-system
- ktl get [xyz] -o wide
	- node
	- svc
	- hpa
- kubectl port-forward service/nginx-service 30080:80
	- port forward
- horizontal scaling
	- kubectl autoscale deployment php-apache --cpu-percent=50 --min=1 --max=10
		- add pod when cpu is less than 50%.  It will add P pods till it meets target utilization
		- ktl get hpa
		- I believe the 50 is referring to request ==cpu millicore==
- Generate load
	- kubectl run -i --tty load-generator --rm --image=busybox:1.28 --restart=Never -- /bin/sh -c "while sleep 0.01; do wget -q -O- http://php-apache; done"
- Watch
	- kubectl get hpa php-apache --watch


### KIND (Instead of Dcoker Desktop)
- using kind - ecosystem faster
	https://kind.sigs.k8s.io/
	faster than minikube
	easier than docker desktop
	BUT networking is harder
		docker does not expost docker network to host
		can access deployment with port forwarding
			forward a host port to a target port inside k8s word
			kubectl port-forward service/nginx-service 30080:80
- Instructions			
		cd samples
		ktl apply -f deployment.yaml
		kubectl port-forward service/nginx-service 30080:80