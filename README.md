# ACBG
This repo contains the official codes for our paper:

### Attentive and Contrastive Image Manipulation Localization With Boundary Guidance
Accepted in IEEE Transactions on Information Forensics and Security.

___
## Testing
To test the model, simply run ```infer.py```. It will probe the images in the ```results``` folder.

> python infer.py
> 
## Evaluting
> python evaluate.py
> 
___
## Training
For training, you need to first download the  training dataset and put them into the ```dataset``` folder correspondingly.

> All file paths about training are stored in ```config.py```. You may modify them to accommodate your system. Besides, the weights of ACBG are stored in the ```ckpt``` folder.

___
## Citations
If ACBG helps your research or work, please kindly cite our paper. The following is a BibTeX reference.
```
@ARTICLE{10589438,
  author={Liu, Wenxi and Zhang, Hao and Lin, Xinyang and Zhang, Qing and Li, Qi and Liu, Xiaoxiang and Cao, Ying},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Attentive and Contrastive Image Manipulation Localization With Boundary Guidance}, 
  year={2024},
  volume={19},
  number={},
  pages={6764-6778},
  keywords={Location awareness;Contrastive learning;Feature extraction;Task analysis;Decoding;Visualization;Deepfakes;Image manipulation detection/localization},
  doi={10.1109/TIFS.2024.3424987}}
```


