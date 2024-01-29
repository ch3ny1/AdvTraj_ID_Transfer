import carla, os, queue, cv2
import numpy as np
from itertools import product
from utils.yaml_utils import *
from attack.attack_manager import AttackManager
from tqdm import tqdm

class ReplayManager(AttackManager):
    def __init__(self, scenario_folder=None, settings=None):
        super().__init__(settings)
        self.scenario_folder = scenario_folder
        self.output_folder = None
        self.walker_bp = []

    def replay_trajs(self, v_bp, a_bp, trajs, scenario_name, settings=None, save_imgs=False):
        if settings is None:
            settings = self.settings
        self.setup_carla(settings)
        if trajs.shape[0] == 3:
            self.spawn_ego_vehicle(settings)
            self.ego_vehicle.set_transform(list_to_transform(trajs[2][0]))
            self.spawn_camera(settings)
            image_size_y = settings['camera']['image_size_y']
            image_size_x = settings['camera']['image_size_x']
        else:
            self.spawn_surveillance_camera(settings)
            image_size_y = settings['surveillance_camera']['image_size_y']
            image_size_x = settings['surveillance_camera']['image_size_x']
        settings['victim']['blueprint'] = 'walker.pedestrian.' + str(v_bp)
        settings['attacker']['blueprint'] = 'walker.pedestrian.' + str(a_bp)
        self.spawn_attacker(settings)
        self.spawn_victim(settings)
        self.victim_walker.set_transform(list_to_transform(trajs[0][0]))
        self.attacker_walker.set_transform(list_to_transform(trajs[1][0]))

        # Store images
        image_queue = queue.Queue()
        self.camera_actor.listen(image_queue.put)
        

        count = 0

        for t in range(1,trajs.shape[1]):
            self.world.tick()
            if save_imgs:
                img_dir = os.path.join(self.output_folder, '{}_{}'.format(v_bp, a_bp), scenario_name)
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                while image_queue.empty() == True:
                    continue
                img = image_queue.get()
                img = np.reshape(np.copy(img.raw_data), (image_size_y, image_size_x, 4))
                cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(count)), img)
                count += 1
            self.victim_walker.set_transform(list_to_transform(trajs[0][t]))
            self.attacker_walker.set_transform(list_to_transform(trajs[1][t]))
            if trajs.shape[0] == 3:
                self.ego_vehicle.set_transform(list_to_transform(trajs[2][t]))
        self.cleanup()
        


    def collect_new_imgs(self, walker_bp=None):
        if walker_bp is not None:
            self.walker_bp = walker_bp
        bp_comb = product(self.walker_bp, self.walker_bp)
        #self.setup_carla()
        #bp_library = self.world.get_blueprint_library()
        for v_bp, a_bp in tqdm(bp_comb):
            #v_bp = bp_library.filter('walker.pedestrian.' + str(v_bp))[0]
            #a_bp = bp_library.filter('walker.pedestrian.' + str(a_bp))[0]
            scenarios = os.listdir(self.scenario_folder)
            img_dir = os.path.join(self.output_folder, '{}_{}'.format(v_bp, a_bp))
            if os.path.exists(img_dir) and len(os.listdir(img_dir))==len(scenarios):
                continue
            for scenario in scenarios:
                trajs = np.load(os.path.join(self.scenario_folder, scenario))
                scenario_name = scenario.split('.')[0]
                self.replay_trajs(v_bp, a_bp, trajs, scenario_name, save_imgs=True)