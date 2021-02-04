from generate_gauss_obj import generate_obj_from_file
import os
import glob

obj_generator_list={"watershed": generate_obj_from_file,
                    "voronoi": generate_voronoi_from_file}

def generate_obj(data_dir, obj_method=None):
    if obj_method is None:
        obj_generator = obj_generator_list["watershed"]
    else:
        obj_generator = obj_generator_list[obj_method]

    for norm_img_file in glob.glob(os.path.join(data_dir, "Norms")):
        norm_img_ = os.path.basename(norm_img_file)
        norm_img_ =


def preprocess(data_dir, normalizer=None, obj_method=None):
    if not os.path.exists(os.path.join(data_dir, "Norms")):
        stainNorm(data_dir, normalizer=normalizer)

    if not os.path.exists(os.path.join(data_dir, "Objs")):
        generate_obj(data_dir, obj_method=obj_method)