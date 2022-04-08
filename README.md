# Roadmap
## DATASET:
	1. (DONE) dataset creation
	
## MODEL:
	2. test different GAN architectures for the sim2real translation:
	    [x] CycleGAN
	    [ ] Cycada
	    ...
	    2.1 hyperparameter optimization for each model
	        CycleGAN
	        Cycada
	        ...
            DATASET: Chris1/sim2real_gta5_to_cityscapes 
                (unpaired image-translation dataset)
             	
             	to read it within a python script or notebook,
             	simply do the following
             	
             	
             	```{python}
                from datasets import load_dataset
                dataset = load_dataset("Chris1/sim2real_gta5_to_cityscapes")  
                ```  
## VALIDATION
	3. group results of the tested GANs in a benchmark table, with respect to a common metric (examples are https://github.com/huggingface/community-events/tree/main/huggan/pytorch/metrics)

## HUGGAN SPRINT EVALUATION:
	4. upload models with huggingface specification
	5. showcase results in SPACE demo Huggingface
	    5.1  create model card
	    5.2  create SPACE card and demo



## BONUS TRACK: downstream SEGMENTATION RESULTS:
	6. (optional) benchmark the found GANs with the purpose of semantic segmentation in the real domain 
	       e.g. synthetic images are translated to real and used to train one or more semantic seg models with supervision on the translated data
	    DATASETS: Chris1/cityscapes (the full cityscapes dataset, with train, validation and test splits)
	                               images are paired with the associated semantic seg
	                               (CARE, the test set does not have ground truths, as it is used to produce predictions 
	                                    for the cityscapes evaluation server)
	                                    
	             Chris1/GTA5 (the full GTA5 dataset, with train, validation and test splits)
	                               images are paired with the associated semantic seg masks	