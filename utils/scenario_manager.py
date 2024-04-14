import carla, os, queue, cv2
from collections import OrderedDict
from utils.yaml_utils import *
from utils.bbox_util import build_projection_matrix, get_2d_bbox, get_world_point

class ScenarioManager(object):
    def __init__(self, settings):
        self.settings = settings
        self.world = None
        self.spectator = None
        self.camera_actor = None
        self.proj_mat = None
        self.proj_mat_inv = None
        self.victim_walker = None
        self.attacker_walker = None
        self.ego_vehicle = None
        self.actor_list = []

    def setup_carla(self, settings=None):
        if settings is None:
            settings = self.settings
        client = carla.Client(settings['simulation']['host'], settings['simulation']['port'])
        carla_world = client.get_world()  
        client.load_world(settings['world']['map'])
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

    def collect_bbox(self, actor):
        world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
        bbox = get_2d_bbox(actor, self.proj_mat, world_2_camera)
        return bbox

    def cleanup(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
        self.attacker_walker = None
        self.victim_walker = None
        self.camera_actor = None
        self.ego_vehicle = None

    def run(self, collect_bbox=True, find_attacker_init=False, settings=None):
        if settings is None:
            settings = self.settings
        else:
            self.setup_carla(settings)
        if 'ego_vehicle' in settings:
            self.spawn_ego_vehicle()
        self.spawn_camera(settings)
        self.spawn_victim(settings)
        bboxes = []
        if find_attacker_init == True:
            self.spawn_attacker()
            self.world.tick()
            spectator_transform = self.camera_actor.get_transform()
            self.spectator.set_transform(spectator_transform)
            attacker_bbox = self.collect_bbox(self.attacker_walker)
            self.cleanup()
            return attacker_bbox
        self.world.tick()
        victim_control = list_to_walkercontrol(settings['victim']['init_speed'])
        if self.ego_vehicle is not None:
            ego_vehicle_speed = list_to_vector3D(settings['ego_vehicle']['init_speed'])
            self.ego_vehicle.enable_constant_velocity(ego_vehicle_speed)
        for i in range(settings['simulation']['max_frame']):
            spectator_transform = self.camera_actor.get_transform()
            self.spectator.set_transform(spectator_transform)
            self.victim_walker.apply_control(victim_control)
            if collect_bbox:
                bboxes.append(self.collect_bbox(self.victim_walker))
            self.world.tick()
        self.cleanup()
        if collect_bbox:
            return bboxes

    def run_traj(self, attacker_traj, switched_idx=1000, img_dir=None, bbox_dir=None, settings=None):
        """
        Given the attacker_traj, generated using the collected victim bboxes
        The settings used to generate the victim bboxes
        Replay the scenario while map the attacker back to the scene (bbox -> carla location)
        Also calculate the intra-frame speed needed for the attacker pedestrian and move accordingly
        """
        if settings is None:
            settings = self.settings
        else:
            self.setup_carla(settings)
        if 'ego_vehicle' in settings:
            self.spawn_ego_vehicle()
        self.spawn_camera(settings)
        self.spawn_victim(settings)
        image_queue = queue.Queue()
        #output_image_queue = queue.Queue()
        self.camera_actor.listen(image_queue.put)
        self.spawn_attacker(settings)
        self.world.tick()

        # Save bboxes
        world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
        bboxes = [[get_2d_bbox(self.victim_walker, self.proj_mat, world_2_camera)], \
                  [get_2d_bbox(self.attacker_walker, self.proj_mat, world_2_camera)]] # [[victim], [attacker]]
        # We need the z-coordinate of the attacker to later map 2D bbox to 3D location uniquely
        attacker_z = self.attacker_walker.get_location().z
        victim_control = list_to_walkercontrol(settings['victim']['init_speed'])

        if self.ego_vehicle is not None:
            ego_vehicle_speed = list_to_vector3D(settings['ego_vehicle']['init_speed'])
            self.ego_vehicle.enable_constant_velocity(ego_vehicle_speed)

        # Iterate over all bbox, collect the images and retrieve bboxes from Carla
        for i in range(1, settings['simulation']['max_frame']):
            spectator_transform = self.camera_actor.get_transform()
            self.spectator.set_transform(spectator_transform)
            self.victim_walker.apply_control(victim_control)
            # Matrices needed to find the world location
            camera_2_world = np.array(self.camera_actor.get_transform().get_matrix())
            if i <= switched_idx:
                attacker_target_bbox = attacker_traj[i]
                attacker_image_point = [(attacker_target_bbox[0]+attacker_target_bbox[2])/2,\
                                        (attacker_target_bbox[1]+attacker_target_bbox[3])/2]
                attacker_world_point = get_world_point(attacker_image_point, self.proj_mat_inv, camera_2_world,\
                                                    self.camera_actor.get_location(), attacker_z)
                # Calculate the movement required by the attacker to move to that location
                #self.attacker_walker.set_location(attacker_world_point) # Try teleporting to see what's happening
                attacker_curr_location = self.attacker_walker.get_location()
                direction = attacker_world_point - attacker_curr_location
                speed = attacker_world_point.distance(attacker_curr_location) / settings['world']['fixed_delta_seconds']
                #print("Speed (m/s):" + str(speed))
                attacker_control = WalkerControl(direction, speed)
                self.attacker_walker.apply_control(attacker_control)
            self.world.tick()
            #if img_dir is not None:
            #    image = image_queue.get()
            #    image.save_to_disk(os.path.join(img_dir, ''))
                #img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
                #output_image_queue.put(img)
            
            # Retrieve bbox from Carla (bboxes we actually achieved)
            world_2_camera = np.array(self.camera_actor.get_transform().get_inverse_matrix())
            victim_bbox = get_2d_bbox(self.victim_walker, self.proj_mat, world_2_camera)
            attacker_bbox = get_2d_bbox(self.attacker_walker, self. proj_mat, world_2_camera)
            bboxes[0].append(victim_bbox)
            bboxes[1].append(attacker_bbox)
        
        if bbox_dir is not None:
            np.save(os.path.join(bbox_dir, 'bboxes.npy'), np.array(bboxes))
        
        if img_dir is not None:
            count = 0
            print("Saving images")
            while image_queue.empty() != True:
                img = image_queue.get()
                img.save_to_disk(os.path.join(img_dir, '{}.jpg'.format(count)))
                count += 1
        
        self.cleanup()


