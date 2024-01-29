import carla, os, queue, cv2, random
import tensorflow as tf
from collections import OrderedDict
from utils.yaml_utils import *
#from utils.bbox_util import build_projection_matrix, get_2d_bbox, get_world_point, get_2d_bbox_with_shift, get_verts
from utils.bbox_util import *
from ID_Transfer.ID_Transfer import *
from ID_Transfer.sort_TF import *
from datetime import datetime

class AttackManager(object):
    def __init__(self, settings):
        self.settings = settings
        self.world = None
        self.fixed_delta_seconds = settings['world']['fixed_delta_seconds']
        self.spectator = None
        self.camera_actor = None
        self.proj_mat = None
        self.proj_mat_inv = None
        self.victim_walker = None
        self.attacker_walker = None # The target object
        self.ego_vehicle = None
        self.actor_list = []
        # For optimization
        self.threshold = 0.1
        self.attacker_speed_limit = 3
        self.attacker_movement_limit = self.attacker_speed_limit * self.fixed_delta_seconds
        self.lr = settings['simulation']['lr']
        self.iter = settings['simulation']['iter']

    def setup_carla(self, settings=None):
        if settings is None:
            settings = self.settings
        client = carla.Client(settings['simulation']['host'], settings['simulation']['port'])
        carla_world = client.get_world()  
        client.load_world(settings['world']['map'], carla.MapLayer.Buildings)
        world_settings = carla_world.get_settings()
        if not world_settings.synchronous_mode:
            world_settings.synchronous_mode = True
            world_settings.fixed_delta_seconds = settings['world']['fixed_delta_seconds']
            carla_world.apply_settings(world_settings)
        self.world = carla_world
        self.spectator = carla_world.get_spectator()

    def spawn_ego_vehicle(self, settings=None):
        if settings is None:
            settings = self.settings
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.filter(settings['ego_vehicle']['blueprint'])[0]
        vehicle_transform = list_to_transform(settings['ego_vehicle']['spawn_transform'])
        vehicle_actor = self.world.spawn_actor(vehicle_bp, vehicle_transform)
        self.actor_list.append(vehicle_actor)
        self.ego_vehicle = vehicle_actor


    def spawn_camera(self, settings=None):
        if settings is None:
            settings = self.settings
        bp_library = self.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        image_w = settings['camera']['image_size_x']
        image_h = settings['camera']['image_size_y']
        fov = settings['camera']['fov']
        camera_bp.set_attribute('image_size_x', str(image_w))
        camera_bp.set_attribute('image_size_y', str(image_h))
        camera_bp.set_attribute('fov', str(fov))
        if self.ego_vehicle is not None:
            camera_spawn_point = list_to_transform(settings['camera']['attach_transform'])
            camera_actor = self.world.spawn_actor(camera_bp, camera_spawn_point, attach_to=self.ego_vehicle)
        else:
            camera_spawn_point = list_to_transform(settings['camera']['transform'])
            camera_actor = self.world.spawn_actor(camera_bp, camera_spawn_point)
        self.camera_actor = camera_actor
        self.actor_list.append(camera_actor)
        self.proj_mat = build_projection_matrix(image_w, image_h, fov)
        self.proj_mat_inv = np.linalg.inv(self.proj_mat)
             

    def spawn_victim(self, settings=None):
        if settings is None:
            settings = self.settings
        victim_bp = self.world.get_blueprint_library().find(settings['victim']['blueprint'])
        victim_spawn_point = list_to_transform(settings['victim']['spawn_transform'])
        victim_actor = self.world.spawn_actor(victim_bp, victim_spawn_point)
        self.victim_walker = victim_actor
        self.actor_list.append(victim_actor)

    def spawn_attacker(self, settings=None):
        if settings is None:
            settings = self.settings
        attacker_bp = self.world.get_blueprint_library().find(settings['attacker']['blueprint'])
        attacker_spawn_point = list_to_transform(settings['attacker']['spawn_transform'])
        attacker_actor = self.world.spawn_actor(attacker_bp, attacker_spawn_point)
        self.attacker_walker = attacker_actor
        self.actor_list.append(attacker_actor)

    def spawn_surveillance_camera(self, settings=None):
        if settings is None:
            settings = self.settings
        bp_library = self.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        image_w = settings['surveillance_camera']['image_size_x']
        image_h = settings['surveillance_camera']['image_size_y']
        fov = settings['surveillance_camera']['fov']
        camera_bp.set_attribute('image_size_x', str(image_w))
        camera_bp.set_attribute('image_size_y', str(image_h))
        camera_bp.set_attribute('fov', str(fov))
        camera_spawn_point = list_to_transform(settings['surveillance_camera']['transform'])
        camera_actor = self.world.spawn_actor(camera_bp, camera_spawn_point)
        self.camera_actor = camera_actor
        self.actor_list.append(camera_actor)
        self.proj_mat = build_projection_matrix(image_w, image_h, fov)
        self.proj_mat_inv = np.linalg.inv(self.proj_mat)
             

    def collect_bbox(self, actor):
        world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
        bbox = get_2d_bbox(actor, self.proj_mat, world_2_camera)
        return bbox
    
    def add_bbox_noise(self, bbox, noise, noise2=None):
        bbox[0] += noise[0]
        bbox[1] += noise[1]
        if noise2 is not None:
            bbox[2] += noise2[0]
            bbox[3] += noise2[1]
        else:
            bbox[2] += noise[0]
            bbox[3] += noise[1]
        return bbox

    def cleanup(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        self.attacker_walker = None
        self.victim_walker = None
        self.camera_actor = None
        self.ego_vehicle = None

    def run_attack_3d(self, settings=None, higher_view = False, save_imgs_bboxes = False):
        if settings is None:
            settings = self.settings
        else:
            self.setup_carla(settings)
        # Spawn the actors
        self.spawn_ego_vehicle(settings)
        self.spawn_camera(settings)
        self.spawn_victim(settings)
        self.spawn_attacker(settings)
        spectator_transform = self.camera_actor.get_transform()
        if higher_view:
            spectator_transform.location.z = spectator_transform.location.z + 3.
            spectator_transform.rotation.pitch = -20.
        self.spectator.set_transform(spectator_transform)
        attacker = self.attacker_walker
        attacker.set_simulate_physics(False)
        victim = self.victim_walker

        # Store images
        image_queue = queue.Queue()
        self.camera_actor.listen(image_queue.put)

        self.world.tick()

        # Save bboxes
        world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
        bboxes = [[get_2d_bbox(self.victim_walker, self.proj_mat, world_2_camera)], \
                  [get_2d_bbox(self.attacker_walker, self.proj_mat, world_2_camera)]] # [[victim], [attacker]]
        
        # Store trajectories
        victim_traj = [transform_to_list(victim.get_transform())]
        attacker_traj = [transform_to_list(attacker.get_transform())]
        ego_traj = [transform_to_list(self.ego_vehicle.get_transform())]

        # Movement for the ego vehicle and victim walker
        if self.ego_vehicle is not None:
            ego_vehicle_speed = list_to_vector3D(settings['ego_vehicle']['init_speed'])
            self.ego_vehicle.enable_constant_velocity(ego_vehicle_speed)
        victim_control = list_to_walkercontrol(settings['victim']['init_speed'])

        # Record some useful constants
        victim_delta_location = victim_control.direction * victim_control.speed * settings['world']['fixed_delta_seconds']
        #map = self.world.get_map()

        # Initialize KF trackers for attack
        victim_tracker = KalmanBoxTracker(bboxes[0][0])
        attacker_tracker = KalmanBoxTracker(bboxes[1][0])

        # Create variable for optimization
        #attacker_states = tf.Variable(convert_bbox_to_z(bboxes[1][0])[:3])
        switched = False

        # Start the attack
        for i in range(1, settings['simulation']['max_frame']):
            victim_traj.append(transform_to_list(victim.get_transform()))
            attacker_traj.append(transform_to_list(attacker.get_transform()))
            ego_traj.append(transform_to_list(self.ego_vehicle.get_transform()))
            spectator_transform = self.camera_actor.get_transform()
            if higher_view:
                spectator_transform.location.z = spectator_transform.location.z + 3.
                spectator_transform.rotation.pitch = -20.
            self.spectator.set_transform(spectator_transform)
            self.victim_walker.apply_control(victim_control)

            # Victim KF predicted bbox at t
            victim_pred = victim_tracker.predict_no_trace()[0]
            # Attacker KF predicted bbox at t
            attakcer_pred = attacker_tracker.predict_no_trace()[0]
            # Get current w2c
            vehicle_delta_location = self.ego_vehicle.get_velocity() * settings['world']['fixed_delta_seconds']
            camera_transform = self.camera_actor.get_transform()
            camera_transform.location = camera_transform.location + vehicle_delta_location
            world_2_camera_curr = tf.constant(camera_transform.get_inverse_matrix(), dtype=tf.float32)
            # Victim actual bbox at t
            victim_noise = [np.random.normal(0., .5), np.random.normal(0., 1.)]
            #victim_noise = [0., 0.]
            victim_curr = get_2d_bbox_with_shift(victim, self.proj_mat, world_2_camera_curr, victim_delta_location)
            #print(victim_curr)
            victim_curr = self.add_bbox_noise(victim_curr, victim_noise)
            #print("perturbed: {}".format(victim_curr))
            
            # Attacker actual bbox at t
            attacker_curr = bboxes[1][-1]
            ious = iou_batch([victim_curr, attacker_curr], [victim_pred, attakcer_pred])
            # For debug and see results in numerical fashion
            #print(ious)
            if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0] or switched: # ID already switched
                switched = True
                attacker.apply_control(WalkerControl(Vector3D(0,0,0), 0))

                self.world.tick()

                world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
                victim_bbox = get_2d_bbox(victim, self.proj_mat, world_2_camera)
                victim_bbox = self.add_bbox_noise(victim_bbox, victim_noise)
                attacker_bbox = get_2d_bbox(attacker, self.proj_mat, world_2_camera)
                bboxes[0].append(victim_bbox)
                bboxes[1].append(attacker_bbox)
                victim_tracker.predict()
                attacker_tracker.predict()
                victim_tracker.update(attacker_bbox)
                attacker_tracker.update(victim_bbox)
            else:
                # Attacker 3D bbox (8 vertices) at t
                attacker_3d_bbox = get_verts(attacker)
                # How far we should move
                center_displacement = optimize_3d_coord(attacker_tracker, victim_tracker, attacker_3d_bbox, victim_curr, \
                                                        self.proj_mat, world_2_camera_curr, lr=settings['simulation']['lr'], \
                                                        iteration=settings['simulation']['iter'], delta_location=self.attacker_movement_limit)
                #print(center_displacement)
                direction = carla.Vector3D(center_displacement[0].item(), center_displacement[1].item(), 0.)
                speed = np.linalg.norm(center_displacement) / self.fixed_delta_seconds
                if speed > self.attacker_speed_limit:
                    speed = self.attacker_speed_limit
                #attacker_control = carla.WalkerControl(direction, speed)
                target_location = attacker.get_location() + direction
                target_location.z = attacker.get_location().z
                # Enforce that the walker is on the sidewalk
                #target_waypoint = map.get_waypoint(target_location, project_to_road=False, \
                #                                   lane_type=carla.LaneType.Sidewalk)
                #if target_waypoint is None:
                #    target_waypoint = map.get_waypoint(target_location, project_to_road=True, \
                #                                   lane_type=carla.LaneType.Sidewalk)
                #target_location = target_waypoint.transform.location
                attacker.set_location(target_location)
                #print(attacker_control)
                #attacker.apply_control(attacker_control)

                self.world.tick()
                #print(victim.get_location())
                #print(attacker.get_location())
                #print(attacker.get_velocity())

                world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
                victim_bbox = get_2d_bbox(victim, self.proj_mat, world_2_camera)
                victim_bbox = self.add_bbox_noise(victim_bbox, victim_noise)
                attacker_bbox = get_2d_bbox(attacker, self.proj_mat, world_2_camera)
                bboxes[0].append(victim_bbox)
                bboxes[1].append(attacker_bbox)
                victim_tracker.predict()
                attacker_tracker.predict()
                victim_tracker.update(victim_bbox)
                attacker_tracker.update(attacker_bbox)

        
        # Write image files and bboxes to disk
        if switched and save_imgs_bboxes==True:
            current_time = datetime.now()
            current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
            scenario_name = current_time
            img_dir = os.path.join(settings['output_settings']['img_dir'], scenario_name)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            bbox_dir = settings['output_settings']['bbox_dir']
            config_dir = settings['output_settings']['config_dir']
            #config_dir = r'D:\ID-Switch-Sim-Output\roadside_diff\configs'
            traj_dir = settings['output_settings']['traj_dir']

            np.save(os.path.join(traj_dir, '{}.npy'.format(scenario_name)), np.array([victim_traj, attacker_traj, ego_traj]))
            np.save(os.path.join(bbox_dir, '{}.npy'.format(scenario_name)), np.array(bboxes))
            save_yaml(settings, os.path.join(config_dir, '{}.yaml'.format(scenario_name)))
            count = 0
            print("Saving images")
            while image_queue.empty() != True:
                img = image_queue.get()
                #img.save_to_disk(os.path.join(img_dir, '{}.jpg'.format(count)))
                img = np.reshape(np.copy(img.raw_data), (settings['camera']['image_size_y'], settings['camera']['image_size_x'], 4))
                cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(count)), img)
                count += 1
        self.cleanup()
        return switched

    def run_attack_3d_surveillance(self, settings=None, higher_view = False, save_imgs_bboxes = False):
        if settings is None:
            settings = self.settings
        else:
            self.setup_carla(settings)
        # Spawn the actors
        if 'surveillance_camera' in settings:
            self.spawn_surveillance_camera(settings)
        self.spawn_victim(settings)
        self.spawn_attacker(settings)
        spectator_transform = self.camera_actor.get_transform()
        if higher_view:
            spectator_transform.location.z = spectator_transform.location.z + 3.
            spectator_transform.rotation.pitch = -20.
        self.spectator.set_transform(spectator_transform)
        attacker = self.attacker_walker
        attacker.set_simulate_physics(False)
        victim = self.victim_walker

        # Store images
        image_queue = queue.Queue()
        self.camera_actor.listen(image_queue.put)


        self.world.tick()

        # Save bboxes
        world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
        bboxes = [[get_2d_bbox(self.victim_walker, self.proj_mat, world_2_camera)], \
                  [get_2d_bbox(self.attacker_walker, self.proj_mat, world_2_camera)]] # [[victim], [attacker]]
        
        # Store trajectories
        victim_traj = [transform_to_list(victim.get_transform())]
        attacker_traj = [transform_to_list(attacker.get_transform())]

        # Movement for the ego vehicle and victim walker
        victim_control = list_to_walkercontrol(settings['victim']['init_speed'])

        # Record some useful constants
        victim_delta_location = victim_control.direction * victim_control.speed * settings['world']['fixed_delta_seconds']
        #map = self.world.get_map()

        # Initialize KF trackers for attack
        victim_tracker = KalmanBoxTracker(bboxes[0][0])
        attacker_tracker = KalmanBoxTracker(bboxes[1][0])

        # Create variable for optimization
        #attacker_states = tf.Variable(convert_bbox_to_z(bboxes[1][0])[:3])
        switched = False

        # Start the attack
        for i in range(1, settings['simulation']['max_frame']):
            victim_traj.append(transform_to_list(victim.get_transform()))
            attacker_traj.append(transform_to_list(attacker.get_transform()))
            spectator_transform = self.camera_actor.get_transform()
            if higher_view:
                spectator_transform.location.z = spectator_transform.location.z + 3.
                spectator_transform.rotation.pitch = -20.
            self.spectator.set_transform(spectator_transform)
            self.victim_walker.apply_control(victim_control)

            # Victim KF predicted bbox at t
            victim_pred = victim_tracker.predict_no_trace()[0]
            # Attacker KF predicted bbox at t
            attakcer_pred = attacker_tracker.predict_no_trace()[0]
            # Get current w2c
            camera_transform = self.camera_actor.get_transform()
            world_2_camera_curr = tf.constant(camera_transform.get_inverse_matrix(), dtype=tf.float32)
            # Victim actual bbox at t
            victim_noise = [np.random.normal(0., .5), np.random.normal(0., 1.)]
            victim_curr = get_2d_bbox_with_shift(victim, self.proj_mat, world_2_camera_curr, victim_delta_location)
            #print(victim_curr)
            victim_curr = self.add_bbox_noise(victim_curr, victim_noise)
            #print("perturbed: {}".format(victim_curr))
            
            # Attacker actual bbox at t
            attacker_curr = bboxes[1][-1]
            ious = iou_batch([victim_curr, attacker_curr], [victim_pred, attakcer_pred])
            # For debug and see results in numerical fashion
            #print(ious)
            if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0] or switched: # ID already switched
                switched = True
                attacker.apply_control(WalkerControl(Vector3D(0,0,0), 0))

                self.world.tick()

                world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
                victim_bbox = get_2d_bbox(victim, self.proj_mat, world_2_camera)
                victim_bbox = self.add_bbox_noise(victim_bbox, victim_noise)
                attacker_bbox = get_2d_bbox(attacker, self.proj_mat, world_2_camera)
                bboxes[0].append(victim_bbox)
                bboxes[1].append(attacker_bbox)
                victim_tracker.predict()
                attacker_tracker.predict()
                victim_tracker.update(attacker_bbox)
                attacker_tracker.update(victim_bbox)
            else:
                # Attacker 3D bbox (8 vertices) at t
                attacker_3d_bbox = get_verts(attacker)
                # How far we should move
                center_displacement = optimize_3d_coord(attacker_tracker, victim_tracker, attacker_3d_bbox, victim_curr, \
                                                        self.proj_mat, world_2_camera_curr, lr=settings['simulation']['lr'], \
                                                        iteration=settings['simulation']['iter'], delta_location=self.attacker_movement_limit)
                #print(center_displacement)
                direction = carla.Vector3D(center_displacement[0].item(), center_displacement[1].item(), 0.)
                speed = np.linalg.norm(center_displacement) / self.fixed_delta_seconds
                if speed > self.attacker_speed_limit:
                    speed = self.attacker_speed_limit
                #attacker_control = carla.WalkerControl(direction, speed)
                target_location = attacker.get_location() + direction
                target_location.z = attacker.get_location().z
                attacker.set_location(target_location)

                self.world.tick()

                world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
                victim_bbox = get_2d_bbox(victim, self.proj_mat, world_2_camera)
                victim_bbox = self.add_bbox_noise(victim_bbox, victim_noise)
                attacker_bbox = get_2d_bbox(attacker, self.proj_mat, world_2_camera)
                bboxes[0].append(victim_bbox)
                bboxes[1].append(attacker_bbox)
                victim_tracker.predict()
                attacker_tracker.predict()
                victim_tracker.update(victim_bbox)
                attacker_tracker.update(attacker_bbox)

        self.cleanup()
        # Write image files and bboxes to disk
        if switched and save_imgs_bboxes==True:
            current_time = datetime.now()
            current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
            scenario_name = current_time
            img_dir = os.path.join(settings['output_settings']['img_dir'], scenario_name)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            bbox_dir = settings['output_settings']['bbox_dir']
            config_dir = settings['output_settings']['config_dir']
            traj_dir = settings['output_settings']['traj_dir']
            #config_dir = r'D:\ID-Switch-Sim-Output\surveillance_diff\configs'

            np.save(os.path.join(bbox_dir, '{}.npy'.format(scenario_name)), np.array(bboxes))
            np.save(os.path.join(traj_dir, '{}.npy'.format(scenario_name)), np.array([victim_traj, attacker_traj]))
            save_yaml(settings, os.path.join(config_dir, '{}.yaml'.format(scenario_name)))
            count = 0
            print("Saving images")
            while image_queue.empty() != True:
                img = image_queue.get()
                #img.save_to_disk(os.path.join(img_dir, '{}.jpg'.format(count)))
                img = np.reshape(np.copy(img.raw_data), (settings['surveillance_camera']['image_size_y'], settings['surveillance_camera']['image_size_x'], 4))
                cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(count)), img)
                count += 1

        return switched
    

    def collect_baseline(self, settings=None, higher_view=False, save_img_bbox=False):
        """
        Victim and attacker are both constant speed actors having intersecting trajectories
        """
        if settings is None:
            settings = self.settings
        else:
            self.setup_carla(settings)
        # Spawn the actors
        if 'surveillance_camera' in settings:
            self.spawn_surveillance_camera(settings)
        self.spawn_victim(settings)
        # Get randomized spawn point for another pedestrian (using the attacker actor)
        # that has intersecting trajectory with the victim to use as baseline
        self.spawn_attacker(settings)
        spectator_transform = self.camera_actor.get_transform()
        if higher_view:
            spectator_transform.location.z = spectator_transform.location.z + 3.
            spectator_transform.rotation.pitch = -20.
        self.spectator.set_transform(spectator_transform)
        attacker = self.attacker_walker
        victim = self.victim_walker
        # Store images
        image_queue = queue.Queue()
        self.camera_actor.listen(image_queue.put)

        self.world.tick()

        # Save bboxes
        world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
        bboxes = [[get_2d_bbox(self.victim_walker, self.proj_mat, world_2_camera)], \
                  [get_2d_bbox(self.attacker_walker, self.proj_mat, world_2_camera)]] # [[victim], [attacker]]

        # Movement for the ego vehicle and victim walker
        victim_control = list_to_walkercontrol(settings['victim']['init_speed'])
        attacker_control = list_to_walkercontrol(settings['attacker']['init_speed'])
        victim_noise = [np.random.normal(0., .5), np.random.normal(0., 1.)] # Noise on pixel level

        # Initialize KF trackers for attack
        victim_tracker = KalmanBoxTracker(bboxes[0][0])
        attacker_tracker = KalmanBoxTracker(bboxes[1][0])

        switched = False

        for i in range(1, settings['simulation']['max_frame']):
            spectator_transform = self.camera_actor.get_transform()
            if higher_view:
                spectator_transform.location.z = spectator_transform.location.z + 3.
                spectator_transform.rotation.pitch = -20.
            self.spectator.set_transform(spectator_transform)
            self.victim_walker.apply_control(victim_control)
            self.attacker_walker.apply_control(attacker_control)

            self.world.tick()
            world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
            victim_bbox = get_2d_bbox(victim, self.proj_mat, world_2_camera)
            #victim_bbox = self.add_bbox_noise(victim_bbox, victim_noise)
            attacker_bbox = get_2d_bbox(attacker, self.proj_mat, world_2_camera)
            bboxes[0].append(victim_bbox)
            bboxes[1].append(attacker_bbox)
            
            victim_pred = victim_tracker.predict()[0]
            attacker_pred = attacker_tracker.predict()[0]

            ious = iou_batch([victim_bbox, attacker_bbox], [victim_pred, attacker_pred])
            if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0] or switched: 
                switched = True
                victim_tracker.update(attacker_bbox)
                attacker_tracker.update(victim_bbox)
            else:
                victim_tracker.update(victim_bbox)
                attacker_tracker.update(attacker_bbox)

            victim_tracker.predict()
            attacker_tracker.predict()
        
        if save_img_bbox:
            current_time = datetime.now()
            current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
            scenario_name = current_time
            img_dir = os.path.join(settings['output_settings']['img_dir'], scenario_name)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            bbox_dir = settings['output_settings']['bbox_dir']
            config_dir = 'C:\\Research\\ID-Switch-Sim-Output\\surveillance_baseline\\configs'

            np.save(os.path.join(bbox_dir, '{}.npy'.format(scenario_name)), np.array(bboxes))
            save_yaml(settings, os.path.join(config_dir, '{}.yaml'.format(scenario_name)))
            count = 0
            print("Saving images")
            while image_queue.empty() != True:
                img = image_queue.get()
                #img.save_to_disk(os.path.join(img_dir, '{}.jpg'.format(count)))
                img = np.reshape(np.copy(img.raw_data), (settings['surveillance_camera']['image_size_y'], settings['surveillance_camera']['image_size_x'], 4))
                cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(count)), img)
                count += 1

        return switched

            
    def run_attack_3d_vehicle(self, settings=None, higher_view = False, save_imgs_bboxes = False):
        if settings is None:
            settings = self.settings
        else:
            self.setup_carla(settings)

        # Remove irrelevant objects
        env_objs = self.world.get_environment_objects(carla.CityObjectLabel.Fences)
        objects_to_toggle = {x.id for x in env_objs}
        self.world.enable_environment_objects(objects_to_toggle, False)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        # Spawn the actors
        self.spawn_ego_vehicle(settings)
        self.spawn_camera(settings)
        # Here they are vehicles
        self.spawn_victim(settings) 
        self.spawn_attacker(settings)

        spectator_transform = self.camera_actor.get_transform()
        if higher_view:
            spectator_transform.location.z = spectator_transform.location.z + 3.
            spectator_transform.rotation.pitch = -20.
        self.spectator.set_transform(spectator_transform)
        attacker = self.attacker_walker
        attacker.set_simulate_physics(False)
        victim = self.victim_walker

        # Store images
        image_queue = queue.Queue()
        self.camera_actor.listen(image_queue.put)

        self.world.tick()

        # Save bboxes
        world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
        bboxes = [[get_2d_bbox(self.victim_walker, self.proj_mat, world_2_camera)], \
                  [get_2d_bbox(self.attacker_walker, self.proj_mat, world_2_camera)]] # [[victim], [attacker]]
        
        # Store trajectories
        victim_traj = [transform_to_list(victim.get_transform())]
        attacker_traj = [transform_to_list(attacker.get_transform())]
        

        # Movement for the target vehicle
        #victim_control = list_to_walkercontrol(settings['victim']['init_speed'])
        victim_speed = carla.Vector3D(settings['victim']['init_speed'], 0, 0)
        victim.enable_constant_velocity(victim_speed)

        # Record some useful constants
        victim_direction = victim.get_transform().get_forward_vector()
        victim_delta_location = victim_direction.dot(victim_speed) \
            * settings['world']['fixed_delta_seconds'] * victim_direction
        #map = self.world.get_map()

        # Initialize KF trackers for attack
        victim_tracker = KalmanBoxTracker(bboxes[0][0])
        attacker_tracker = KalmanBoxTracker(bboxes[1][0])

        # Create variable for optimization
        #attacker_states = tf.Variable(convert_bbox_to_z(bboxes[1][0])[:3])
        switched = False

        # Start the attack
        for i in range(1, settings['simulation']['max_frame']):
            victim_traj.append(transform_to_list(victim.get_transform()))
            attacker_traj.append(transform_to_list(attacker.get_transform()))
            spectator_transform = self.camera_actor.get_transform()
            if higher_view:
                spectator_transform.location.z = spectator_transform.location.z + 3.
                spectator_transform.rotation.pitch = -20.
            self.spectator.set_transform(spectator_transform)

            # Victim KF predicted bbox at t
            victim_pred = victim_tracker.predict_no_trace()[0]
            # Attacker KF predicted bbox at t
            attakcer_pred = attacker_tracker.predict_no_trace()[0]
            # Get current w2c
            camera_transform = self.camera_actor.get_transform()
            camera_transform.location = camera_transform.location
            world_2_camera_curr = tf.constant(camera_transform.get_inverse_matrix(), dtype=tf.float32)
            # Victim actual bbox at t
            victim_noise = [np.random.normal(0., 3.), np.random.normal(0., 1.5)]
            noise2 = [np.random.normal(0., 3.), np.random.normal(0., 1.5)]
            victim_curr = get_2d_bbox_with_shift(victim, self.proj_mat, world_2_camera_curr, victim_delta_location)
            victim_curr = self.add_bbox_noise(victim_curr, victim_noise, noise2)
            
            # Attacker actual bbox at t
            attacker_curr = bboxes[1][-1]
            ious = iou_batch([victim_curr, attacker_curr], [victim_pred, attakcer_pred])
            # For debug and see results in numerical fashion
            #print(ious)
            if ious[0,0] + ious[1,1] < ious[0,1] + ious[1,0] or switched: # ID already switched
                switched = True
                attacker.apply_control(carla.VehicleControl(brake=1.0))

                self.world.tick()

                world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
                victim_bbox = get_2d_bbox(victim, self.proj_mat, world_2_camera)
                victim_bbox = self.add_bbox_noise(victim_bbox, victim_noise)
                attacker_bbox = get_2d_bbox(attacker, self.proj_mat, world_2_camera)
                bboxes[0].append(victim_bbox)
                bboxes[1].append(attacker_bbox)
                victim_tracker.predict()
                attacker_tracker.predict()
                victim_tracker.update(attacker_bbox)
                attacker_tracker.update(victim_bbox)
            else:
                # Attacker 3D bbox (8 vertices) at t
                attacker_3d_bbox = get_verts(attacker)
                # How far we should move
                center_displacement = optimize_3d_coord(attacker_tracker, victim_tracker, attacker_3d_bbox, victim_curr, \
                                                        self.proj_mat, world_2_camera_curr, lr=settings['simulation']['lr'], \
                                                        iteration=settings['simulation']['iter'], delta_location=self.attacker_movement_limit)
                #print(center_displacement)
                attacker_direction = attacker.get_transform().get_forward_vector()
                direction = carla.Vector3D(center_displacement[0].item(), center_displacement[1].item(), 0.).dot(attacker_direction) * \
                    attacker_direction
                #speed = np.linalg.norm(center_displacement) / self.fixed_delta_seconds
                #if speed > self.attacker_speed_limit:
                #    speed = self.attacker_speed_limit
                #attacker_control = carla.WalkerControl(direction, speed)
                if attacker_direction.dot(direction) < 0:
                    direction = carla.Vector3D(0,0,0)
                    """
                    switched = True
                    attacker.apply_control(carla.VehicleControl(brake=1.0))

                    self.world.tick()

                    world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
                    victim_bbox = get_2d_bbox(victim, self.proj_mat, world_2_camera)
                    victim_bbox = self.add_bbox_noise(victim_bbox, victim_noise)
                    attacker_bbox = get_2d_bbox(attacker, self.proj_mat, world_2_camera)
                    bboxes[0].append(victim_bbox)
                    bboxes[1].append(attacker_bbox)
                    victim_tracker.predict()
                    attacker_tracker.predict()
                    victim_tracker.update(attacker_bbox)
                    attacker_tracker.update(victim_bbox)
                    """
                elif abs(direction.x) >= self.attacker_speed_limit:
                    direction.x = math.copysign(self.attacker_speed_limit, direction.x)
                target_location = attacker.get_location() + direction
                target_location.z = attacker.get_location().z
                # Enforce that the walker is on the sidewalk
                #target_waypoint = map.get_waypoint(target_location, project_to_road=False, \
                #                                   lane_type=carla.LaneType.Sidewalk)
                #if target_waypoint is None:
                #    target_waypoint = map.get_waypoint(target_location, project_to_road=True, \
                #                                   lane_type=carla.LaneType.Sidewalk)
                #target_location = target_waypoint.transform.location
                attacker.set_location(target_location)
                #print(attacker_control)
                #attacker.apply_control(attacker_control)

                self.world.tick()
                #print(victim.get_location())
                #print(attacker.get_location())
                #print(attacker.get_velocity())

                world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
                victim_bbox = get_2d_bbox(victim, self.proj_mat, world_2_camera)
                victim_bbox = self.add_bbox_noise(victim_bbox, victim_noise)
                attacker_bbox = get_2d_bbox(attacker, self.proj_mat, world_2_camera)
                bboxes[0].append(victim_bbox)
                bboxes[1].append(attacker_bbox)
                victim_tracker.predict()
                attacker_tracker.predict()
                victim_tracker.update(victim_bbox)
                attacker_tracker.update(attacker_bbox)

        self.cleanup()
        # Write image files and bboxes to disk
        if save_imgs_bboxes==True:
            current_time = datetime.now()
            current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
            scenario_name = current_time
            img_dir = os.path.join(settings['output_settings']['img_dir'], scenario_name)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            bbox_dir = settings['output_settings']['bbox_dir']
            config_dir = settings['output_settings']['config_dir']
            traj_dir = settings['output_settings']['traj_dir']

            np.save(os.path.join(bbox_dir, '{}.npy'.format(scenario_name)), np.array(bboxes))
            np.save(os.path.join(traj_dir, '{}.npy'.format(scenario_name)), np.array([victim_traj, attacker_traj]))
            save_yaml(settings, os.path.join(config_dir, '{}.yaml'.format(scenario_name)))
            count = 0
            print("Saving images")
            while image_queue.empty() != True:
                img = image_queue.get()
                #img.save_to_disk(os.path.join(img_dir, '{}.jpg'.format(count)))
                img = np.reshape(np.copy(img.raw_data), (settings['surveillance_camera']['image_size_y'], settings['surveillance_camera']['image_size_x'], 4))
                cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(count)), img)
                count += 1

        return switched