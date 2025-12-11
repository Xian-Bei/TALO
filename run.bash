dataset_ls=(
    waymo
    nuscenes
)
model_ls=(
    VGGT
    # Pi3
    # MapAnything
)
interframe_solver_choice_ls=(
    sim3
    sl4
    tps
)
conf_threshold_ls=(
    60
)
for dataset in ${dataset_ls[@]}; do
    if [ $dataset == "waymo" ]; then
        data_root=Data/$dataset
        save_root=Save/$dataset
        scene_ls=(
            183829460855609442
            3156155872654629090
            3461811179177118163
            4058410353286511411
            5200186706748209867
            6104545334635651714
            16345319168590318167

        )
    fi
    if [ $dataset == "nuscenes" ]; then
        data_root=Data/$dataset
        save_root=Save/$dataset
        scene_ls=(
            scene-0003
            scene-0012
            scene-0013
            scene-0036
            scene-0039
            scene-0092 
            scene-0094 

        )
    fi
    for scene in ${scene_ls[@]}; do
        for conf_threshold in ${conf_threshold_ls[@]}; do
            for model in ${model_ls[@]}; do
                for interframe_solver_choice in ${interframe_solver_choice_ls[@]}; do
                    echo ------------$scene $model $conf_threshold $interframe_solver_choice
                    data_folder=$data_root/$scene
                    save_dir=$model+$conf_threshold+$interframe_solver_choice

                    python main.py --data_folder $data_folder --log_path $save_root/$scene/$save_dir --model $model --conf_threshold $conf_threshold --interframe_solver_choice $interframe_solver_choice  #--vis_map  --port 8089 #--vis_tps #

                    python eval_vis_pcd_traj.py --GT $data_folder --pred $save_root/$scene/$save_dir --eval_pcd --eval_traj --vis

                done
            done
        done
    done
    python summary_traj.py $save_root
    python summary_geom.py $save_root
done