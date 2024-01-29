import os, cv2
from yaml_utils import *
from tqdm import tqdm

def draw_traj(scenario_folder):
    bbox_dir = os.path.join(scenario_folder, 'bboxes')
    output_dir = os.path.join(scenario_folder, 'traj_vis')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    scenarios = os.listdir(bbox_dir)
    for scenario in scenarios:
        image = np.zeros([1080, 1920, 3], np.uint8)
        bboxes = np.load(os.path.join(bbox_dir, scenario))
        centers_x = (bboxes[:, :, 2] + bboxes[:, :, 0]) / 2 
        centers_y = (bboxes[:, :, 3] + bboxes[:, :, 1]) / 2
        #print(centers_x.shape)

        for index in range(centers_x.shape[1]):
            if index == centers_x.shape[1] - 1:
                break
            cv2.line(image, [int(centers_x[0, index]), int(centers_y[0, index])], [int(centers_x[0, index+1]), int(centers_y[0, index+1])], [0, 255, 0], 3) 
            cv2.line(image, [int(centers_x[1, index]), int(centers_y[1, index])], [int(centers_x[1, index+1]), int(centers_y[0, index+1])], [255, 0, 0], 3) 
        
        cv2.imwrite(os.path.join(output_dir, '{}.png'.format(scenario.split('.')[0])), image)

if __name__=="__main__":
    draw_traj(r'E:\ID-Switch-Sim-Output\surveillance_ped_1_10')