# author: Chen Min (minchen@stu.pku.edu.cn)

# command: ./demo.sh example test0

# The DATASET_PATH must contain a folder "images" with all the input images.

# The DATASET_PATH must contain a folder "images" with all the input images.
starttime=$(date +%Y-%m-%d\ %H:%M:%S)

DATASET_PATH0=$1
DATANAME=$2
DATASET_PATH=$DATASET_PATH0/$DATANAME
RESULT_PATH=$DATASET_PATH/result


echo "${DATASET_PATH##*/} 1">> process.txt

#. /home/sstc-pr/anaconda3/etc/profile.d/conda.sh
#conda activate cascade_pl



echo "---resize input images---"
#image_dir:uploda img; image_resize_dir: resized img
#--img_scale: for example depth 500*1000, if img_scale=2, final image 1000*2000

python3 img_process.py  --img_scale 2 --image_dir $DATASET_PATH/images/ --image_resize_dir $DATASET_PATH/colmap/images/

resize_img_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---feature_extractor---"
colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/colmap/images  \
   --ImageReader.camera_model PINHOLE   \
   --SiftExtraction.max_image_size 5200 \
   --ImageReader.single_camera 1

feature_extractor_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---exhaustive_matcher---"
colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

exhaustive_matcher_time=`date +"%Y-%m-%d %H:%M:%S"`

mkdir $DATASET_PATH/colmap/sparse

echo "---mapper---"
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/colmap/images \
    --output_path $DATASET_PATH/colmap/sparse 

mapper_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---bin to txt---"
colmap model_converter \
   --input_path $DATASET_PATH/colmap/sparse/0 \
   --output_path  $DATASET_PATH/colmap/sparse \
   --output_type TXT

bin2txt_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "${DATASET_PATH##*/} 2">> process.txt
echo "---colmap to mvsnet input---"
python3 colmap2mvsnet.py  --dense_folder $DATASET_PATH  --max_d 192  \
--list_folder data/lists/testing_list.txt

colmap2mvsnet_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---cascade_pl---"
########### need to change path: root_dir ckpt_path#############################
python3 cascade_pl_4.12_colmap/eval.py --root_dir ../test_data --depth_dir $RESULT_PATH/depth \
--point_dir $RESULT_PATH/point  --img_scale 2  --save_visual \
--list_dir data/lists/testing_list.txt   --ckpt_path cascade_pl_4.12_colmap/ckpts/exp/_ckpt_epoch_49.ckpt

python3 depth2dmap.py  --root_dir ../test_data --img_scale 2 --depth_dir $RESULT_PATH/depth \
--list_dir data/lists/testing_list.txt

depth_inference_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---OpenMVS---"
cd $RESULT_PATH
echo "---InterfaceCOLMAP---"
/usr/local/bin/OpenMVS/InterfaceCOLMAP -i ../colmap -o scene.mvs 

InterfaceCOLMAP_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---DensifyPointCloud---"
/usr/local/bin/OpenMVS/DensifyPointCloud scene.mvs #--resolution-level 1 #-v 4

DensifyPointCloud_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "${DATASET_PATH##*/} 3">> process.txt
echo "---ReconstructMesh---"
/usr/local/bin/OpenMVS/ReconstructMesh scene_dense.mvs --smooth 5  --d 2 

ReconstructMesh_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---RefineMesh---"
/usr/local/bin/OpenMVS/RefineMesh scene_dense_mesh.mvs --scales 1 --scale-step 1 --resolution-level 2

RefineMesh_time=`date +"%Y-%m-%d %H:%M:%S"`

echo "---TextureMesh---"
echo "${DATASET_PATH##*/} 4">> process.txt
/usr/local/bin/OpenMVS/TextureMesh scene_dense_mesh_refine.mvs --export-type obj --decimate 0.25 --cost-smoothness-ratio 5 

TextureMesh_time=`date +"%Y-%m-%d %H:%M:%S"`

mkdir model

mv scene_dense_mesh_refine_texture.mtl model
mv scene_dense_mesh_refine_texture.obj model
mv scene_dense_mesh_refine_texture_material* model

/usr/local/bin/OpenMVS/TextureMesh scene_dense_mesh_refine.mvs --export-type obj --decimate 0.00390625 --cost-smoothness-ratio 5 

TextureMesh_small_time=`date +"%Y-%m-%d %H:%M:%S"`

mkdir model_small

mv scene_dense_mesh_refine_texture.mtl model_small
mv scene_dense_mesh_refine_texture.obj model_small
mv scene_dense_mesh_refine_texture_material* model_small


mkdir openmvs
mv scene* openmvs


zip -r model.zip model
mv model.zip ../

rm -rf DensifyPointCloud*
rm -rf InterfaceCOLMAP*
rm -rf ReconstructMesh*
rm -rf RefineMesh*
rm -rf TextureMesh*

echo "${DATASET_PATH##*/} 5">> process.txt
Done_time=`date +"%Y-%m-%d %H:%M:%S"`

echo 'start time:' $starttime
echo 'resize_img finished:' $resize_img_time
echo 'feature_extract finished:' $feature_extractor_time
echo 'exhaustive_matcher finished:' $exhaustive_matcher_time
echo 'mapper finished:' $mapper_time
echo 'bin2txt finished:' $bin2txt_time
echo 'colmap2mvsnet finished:' $colmap2mvsnet_time
echo 'depth_inference finished:' $depth_inference_time
echo 'InterfaceCOLMAP finished:' $InterfaceCOLMAP_time
echo 'DensifyPointCloud finished:' $DensifyPointCloud_time
echo 'ReconstructMesh finished:' $ReconstructMesh_time
echo 'RefineMesh finished:' $RefineMesh_time
echo 'TextureMesh finished:' $TextureMesh_time
echo 'TextureMesh_small finished:' $TextureMesh_small_time
echo 'Done!:' $Done_time
