# monocular-depth-estimation

### View our [paper](https://github.com/jmsech/monocular-depth-estimation/blob/main/DL_Final_Project.pdf)

![Alt Text](https://github.com/jmsech/self-supervised-monocular-depth-estimation/blob/master/depth.gif)

To replicate the results from our paper 

`
python3 train.py --epochs 4 --save_file path/to/your/model --save_every 4 --kitti_dir path/to/your/kitti/save --results_file final_model_losses.txt
`

and to evaluate the trained model

`
python3 eval.py --model_path path/to/your/model --kitti_dir path/to/your/kitti/save
`
