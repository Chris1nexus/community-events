# Tasks
Our repository is at [link](https://github.com/Chris1nexus/community-events/blob/main/huggan/pytorch/cyclegan/train.py) 
Quickstart notebook is available at the top level of our repo, and is named sim2real_quickstart.ipynb   
* ##### R&D
   * FID loss
   * cycada implementation
   * choice of the GAN evaluation metrics  [huggingface metrics](https://github.com/huggingface/community-events/tree/main/huggan/pytorch/metrics )
* ##### Hyperparameter optimization
   * add code for logging the metrics and loss at each epoch, to the code [in our repo](https://github.com/Chris1nexus/community-events/blob/main/huggan/pytorch/cyclegan/train.py) 
   * CycleGAN 
       * tune (refer to the [original paper](https://arxiv.org/abs/1703.10593) and the huggan implementation at [link](https://github.com/huggingface/community-events/blob/main/huggan/pytorch/cyclegan/train.py) to better understand their meaning) :
         * learning rate
         * decay_epoch (start decaying the learning rate after the epoch number='decay_epoch' )
         * (optional ?) lambda_id (identity loss weight)
         * (optional ?) lambda_cyc (cycle loss weight)
         * (optional ?) n_residual_blocks
    * Cycada
         * learning rate
         *  ...
* ##### Huggingface SPACE demo development
* ##### (optional, if there is still time after all other tasks) evaluate quality of sim2real on semantic segmentation

# Roadmap
#### DATASET:  
1. [x] dataset creation  
	  
#### MODEL:   
2. test different GAN architectures for the sim2real translation:     
     [x] CycleGAN   
     [ ] Cycada    
     [ ] ...       
2. architecture modifications(list the chosen ones and update when finished):  
     [ ] FID loss CycleGAN  
     [ ] ...     
3. hyperparameter optimization for each model  
     [ ] CycleGAN  
     [ ] Cycada  
     [ ] ...   
   
###### Huggingface Dataset: 
Chris1/sim2real_gta5_to_cityscapes 
(unpaired image-translation dataset)
     	
###### Load the dataset  
To load the dataset within a python script or notebook
simply do the following   
```python
from datasets import load_dataset
dataset = load_dataset("Chris1/sim2real_gta5_to_cityscapes")  
```        
#### VALIDATION
5. [ ]   group results of the tested GANs in a benchmark table, with respect to a common metric, examples are the 
[huggingface metrics](https://github.com/huggingface/community-events/tree/main/huggan/pytorch/metrics )

### HUGGAN SPRINT EVALUATION:
6. * [ ] upload models with huggingface specification    [link](https://github.com/huggingface/community-events/tree/main/huggan#24-model-cards)  
7. * [ ] showcase results in SPACE demo Huggingface   [link](https://github.com/huggingface/community-events/tree/main/huggan#3-create-a-demo)   
7.1. * [ ]  create model card     
7.2. * [ ] create SPACE card and demo     



### BONUS TRACK: downstream SEGMENTATION RESULTS:
* (optional) benchmark the found GANs with the purpose of semantic segmentation in the real domain 
e.g. synthetic images are translated to real and used to train one or more semantic seg models with supervision on the translated data

###### Huggingface Dataset: Chris1/cityscapes
The full cityscapes dataset, with train, validation and test splits.
Images are paired with the associated semantic segmentation
(CARE, the test set does not have ground truths, as it is used to produce predictions 
    for the cityscapes evaluation server)
###### Huggingface Dataset: Chris1/GTA5	                                    
The full GTA5 dataset, with train, validation and test splits. 
Images are paired with the associated semantic segmentation masks	
