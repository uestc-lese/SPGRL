# SPGRL
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
* **dataset**: including \[citeseer, uai, acm, BlogCatalog, flickr, coraml\], required.  
* **labelrate**: including \[20, 40, 60\], required.  

e.g.  
````
python main.py -d citeseer -l 20
````
