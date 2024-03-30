# World Modeling Benchmark Dataset



### This project aims to create a benchmark dataset for tabletop manipulation that extracts action-state pair


Pipeline includes

### stable diffusion image generation --> TripoSR 3D modeling --> Maniskill2 object manipulation pipeline --> output data in json


### Note: segmentation uses
```
# To save
from scipy import sparse
mask_sparse = sparse.csr_matrix(seg_before)
sparse.save_npz('mask_sparse.npz', mask_sparse)

# To Load
from scipy.sparse import load_npz
mask_sparse = load_npz('mask_sparse.npz')
mask = mask_sparse.toarray()
```

# Placements

## 1) Moving objects "forward" / "backward" / "right" / "left"
### moving_one_object.json
    {
        "initial_img": initial/0,
        "result_img": result/0,
        "initial_seg": seg/0,
        "result_seg": seg/1
        "obj_id": [(red cube", 1)]
        "target_object": "red cube",
        "direction": "forward"
    }


## 2) Moving object on another
### moving_object_on_another.json
    {
        "initial_img": initial/0,
        "result_img": result/0,
        "obj_ids": [("blue cube", 1), ("green cube", 2)]
        "initial_object": "blue cube",
        "target_object": "green cube",
        "direction": "front"
    }


## 3) Placing object in between
### placing_object_between.json
    {
        "initial_img": initial/0,
        "result_img": result/0,
        "first_object": "blue cube",
        "between_object": "red cube",
        "second_object": "green cube",
    }

## 4) Remove object
### remove_object.json
    {
        "initial_img": initial/0,
        "result_img": result/0,
        "target_object": "blue cube"
    }



## 5) order by color
### order_by_color.json
    {
        "initial_img": initial/0,
        "result_img": result/0,
        "order": ["blue", "red", "black"]
    }


## 6) order by size
### no json needed

--------- wait for now --------- 
## 7) order by name
### order_by_name.json
    {
        "initial_img": initial/0,
        "result_img": result/0,
        "order": ["red cube", "object"]
    }
--------- wait for now --------- 


## 8) add force toward "left" / "right" / "forward" / "backward"
### add_force.json
    {
        "initial_img": initial/0,
        "result_img": [result/0_1, result/0_2, result/0_3]
        "direction": "left",
        "object": "red cube"
    }


## 9) add force toward another object
### add_force_toward_object.json
    {
        "initial_img": initial/0,
        "result_img": [result/0_1, result/0_2, result/0_3]
        "initial_object": "red cube"
        "target_object": "green cube",
    }

