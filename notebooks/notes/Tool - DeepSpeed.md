<-- Concpets
	- Data Parallel (DP)
		* Input data is de-serialized (aka parallelized) 

		* Input data is atomic at tensor level

		* fed to multiple copies of model


	- Tensor Parallel (TP)
		* Input data tensor is split (non-atmoic) and fed to different gpus


	- Pipeline Parallel (PP)
		* model is split by model layers and distributed to different gpus


	- Sharded DDP 



<-- Implementation: Zero Redundancy Optimizer (ZeRO) 
	* Illustrates DP and TP
	
	* Feature: Support limited gpu memory use case, by offloading to 

	* Data Parallel (DP)
		- shards the tensor like TP, but unlike TP, tensor is re-synced for forward or backward computation

		- input data is deserizlied, and fed parallel to the multple gpus

		- model layers are split/sharded "horizontally" to multiple gpus which enables tensor parallelism

			* Given: 
				La | Lb | Lc
				---|----|---
				a0 | b0 | c0
				a1 | b1 | c1

			* Shard to GPU
				GPU0:
				La | Lb | Lc
				---|----|---
				a0 | b0 | c0

				GPU1:
				La | Lb | Lc
				---|----|---
				a1 | b1 | c1

			* data parallelism
				x0 => GPU0
				
				x1 => GPU1

			* focus on GPU0
				to calculate b0, GPU0 needs a0 and a1. GPU0 has a0, but a1 is on GPU1. When GPU1 finishes with a1, it will sent the result to GPU0.

				It is important that GPUS 
					be on same node
					have fast interconnect, like NVme (vs inter node: NVLink or NVSwich)


<-- Naive Model Parallel (MP) and Pipeline Parallel
	* Aka Vertical Model Parallel (MP)
		- Split consecutive layers of model to a gpu
			===================  ===================
			|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
			===================  ===================
			        gpu0                 gpu1

		- Con: gpu1 is idle while gpu0. Sol: Pipeline parallel

	* Pipleline parallel: chunk incoming batch mciro-batches to create a pipeline, improving gpu utilization
		- see figure on https://huggingface.co/docs/transformers/v4.15.0/parallelism

	* Con: 
		- need to rewrite model to nn.Sequential
		- pipeline api can take onle single/tuple of tensors
		- conditional flow is not possible

