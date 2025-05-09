# DG16M: A Large-Scale Dataset for Dual-Arm Grasping with Force-Optimized Grasps

<a href="">Md Faizal Karim</a>, <a href="">Mohammed Saad Hashmi</a>, <a href="">Shreya Bollimuntha</a>, 
<a href="">Mahesh Reddy Tapeti</a>, <a href="">Gaurav Singh</a>, <a href="">Nagamanikandan Govindan</a>, 
<a href="">K Madhava Krishna</a>

## Installation 

```
pip install -r requirements.txt

cd grasp_generation/meshpy
pip install -e . 
```

## Dataset Generation
We first sample 500 single arm antipodal grasps and create all possible cominations to create dual-arm grasp candidates (along with distance pruning to remove extremely close pairs). They are then passed through the optimizer to find the force-closure valid dual-arm grasp pairs. Finally, we save 2000 valid and 2000 invalid grasp pairs in the dataset. These numbers can be changed in the code for further experimentation. 

```
cd grasp_generation/scripts

python3 generate_dg16m.py --meshes_path <PATH_TO_MESHES> --scaled_meshes <SAVE_PATH> --num_workers 16
``` 

Note: Change the number of CPU workers based on the system. The workers are used to parallelize the grasp generation and then run the CVXPY optimization in parallel using multiprocessing.  

## Visualize

Use <a href="./notebooks/viz_grasps.ipynb">viz_grasps.ipynb</a> to vizualize the grasps. 

## 👏 Acknowledgment

Our codebase in built upon the existing works of <a href="https://github.com/ymxlzgy/DA2">DA2 Dataset</a> and <a href="https://github.com/dkguo/PhyGrasp">PhyGrasp</a>. We thank the authors for releasing the code. 

## 📜 Cite 
```
@article{DG16M,
      title={DG16M: A Large-Scale Dataset for Dual-Arm Grasping with Force-Optimized Grasps}, 
      author={Md Faizal Karim and Mohammed Saad Hashmi and Shreya Bollimuntha and Mahesh Reddy Tapeti and Gaurav Singh and Nagamanikandan Govindan and K Madhava Krishna},
      year={2025},
      eprint={2503.08358},
      url={https://arxiv.org/abs/2503.08358}, 
}
```
