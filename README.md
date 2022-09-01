# Structure-Preserving Graph Representation Learning（**SPGRL**）
> The official code of **ICDM2022** paper: [**Structure-Preserving Graph Representation Learning**]
> <br>Ruiyi Fang, Liangjian Wen, Zhao Kang, Jianzhuang Liu

We propose a new Structure-Preserving Graph Representation Learning method called **SPGRL**. Our main idea is maximizing the mutual information (MI) of the graph structure and feature embedding.

The module is illustrated as follows:

<img src="./images/SPGRL.png" height="450">

# Environment Settings 
* python == 3.8
* Pytorch == 1.1.0  
* Numpy == 1.16.2  
* SciPy == 1.3.1  
* Networkx == 2.4  
* scikit-learn == 0.21.3  

# Usage 
````
python main.py -d dataset -l labelrate
````
* **dataset**: including \[citeseer, uai, acm, BlogCatalog, flickr, cora\], required.  
* **labelrate**: including \[20, 40, 60\], required.  

e.g.  
````
python main.py -d citeseer -l 20
````
