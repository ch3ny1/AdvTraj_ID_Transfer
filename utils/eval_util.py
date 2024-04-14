import itertools, random, copy, math
import numpy as np

# Randomize simulation settings according to specific ranges
class SettingsRandomizer(object):
    def __init__(self, settings = None):
        self.settings = settings
        self.victim_spawn_range = None # [xmin, xmax, ymin, ymax]
        self.attacker_spawn_range = None
        self.victim_speed_range = None
        self.ego_spawn_range = None
        self.ego_speed_range = None # [low, high]
        self.victim_spawn_number = 5
        self.attacker_spawn_number = 5
        self.ego_speed_number = 5

    def generate_settings(self, seed=2023):
        """
        Generate settings according to each scenario in the combination of parameters
        Total number of settings generated:
            victim_spawn_number * attacker_spawn_number * ego_speed_number
        """
        random.seed(seed)
        settings = self.settings
        victim_spawn_z_rotation = settings['victim']['spawn_transform'][2:]
        attacker_spawn_z_rotation = settings['attacker']['spawn_transform'][2:]
        victim_spawn_location = []
        attacker_spawn_location = []
        ego_speed = []
        for i in range(self.victim_spawn_number):
            x = random.uniform(self.victim_spawn_range[0], self.victim_spawn_range[1])
            y = random.uniform(self.victim_spawn_range[2], self.victim_spawn_range[3])
            victim_spawn_location.append([x,y])
        for i in range(self.attacker_spawn_number):
            x = random.uniform(self.attacker_spawn_range[0], self.attacker_spawn_range[1])
            y = random.uniform(self.attacker_spawn_range[2], self.attacker_spawn_range[3])
            attacker_spawn_location.append([x,y])
        for i in range(self.ego_speed_number):
            speed = random.uniform(self.ego_speed_range[0], self.ego_speed_range[1])
            ego_speed.append(speed)
        # Nested loop to create the settings
        settings_list = []
        for victim_loc in victim_spawn_location:
            for attacker_loc in attacker_spawn_location:
                for speed in ego_speed:
                    victim_spawn_transform = victim_loc + victim_spawn_z_rotation
                    attacker_spawn_transform = attacker_loc + attacker_spawn_z_rotation
                    subsettings = copy.deepcopy(settings)
                    subsettings['victim']['spawn_transform'] = victim_spawn_transform
                    subsettings['attacker']['spawn_transform'] = attacker_spawn_transform
                    subsettings['ego_vehicle']['init_speed'] = [speed, 0., 0.]
                    settings_list.append(subsettings)
        
        return settings_list


    def generate_settings_surveillance(self, seed=2023):
        """
        Generate settings according to each scenario in the combination of parameters
        Total number of settings generated:
            victim_spawn_number * attacker_spawn_number * ego_speed_number
        """
        np.random.seed(seed)
        settings = self.settings
        victim_spawn_z_rotation = settings['victim']['spawn_transform'][2:]
        attacker_spawn_z_rotation = settings['attacker']['spawn_transform'][2:]
        victim_spawn_location = []
        attacker_spawn_location = []
        for i in range(self.victim_spawn_number):
            x = np.random.uniform(self.victim_spawn_range[0], self.victim_spawn_range[1])
            y = np.random.uniform(self.victim_spawn_range[2], self.victim_spawn_range[3])
            victim_spawn_location.append([x,y])
        for i in range(self.attacker_spawn_number):
            x = np.random.uniform(self.attacker_spawn_range[0], self.attacker_spawn_range[1])
            y = np.random.uniform(self.attacker_spawn_range[2], self.attacker_spawn_range[3])
            attacker_spawn_location.append([x,y])
        # Nested loop to create the settings
        settings_list = []
        for victim_loc in victim_spawn_location:
            for attacker_loc in attacker_spawn_location:
                victim_spawn_transform = victim_loc + victim_spawn_z_rotation
                attacker_spawn_transform = attacker_loc + attacker_spawn_z_rotation
                subsettings = copy.deepcopy(settings)
                subsettings['victim']['spawn_transform'] = victim_spawn_transform
                subsettings['attacker']['spawn_transform'] = attacker_spawn_transform
                settings_list.append(subsettings)
        
        return settings_list

    def generate_baseline_surveillance(self, speed_limit=(1., 3.), seed=2023):
        """
        Generate settings representing the baseline.
        Victim spawn location is randomized, and the attacker spawn location and speed
        is generated such that its trajectory intersects the victim's trajectory
        """
        np.random.seed(seed)
        settings = self.settings
        victim_spawn_z_rotation = settings['victim']['spawn_transform'][2:]
        attacker_spawn_z_rotation = settings['attacker']['spawn_transform'][2:]
        victim_spawn_location = []
        settings_list = []
        time2intersect = settings['simulation']['max_frame']/2 * settings['world']['fixed_delta_seconds']
        for i in range(self.victim_spawn_number):
            x = np.random.uniform(self.victim_spawn_range[0], self.victim_spawn_range[1])
            y = np.random.uniform(self.victim_spawn_range[2], self.victim_spawn_range[3])
            victim_spawn_location.append([x,y])
        for i in range(self.victim_spawn_number):
            intersecting_point = np.array(victim_spawn_location[i]) + np.array(settings['victim']['init_speed'][:2]) * settings['victim']['init_speed'][3]
            in_range = False
            while in_range == False:
                attacker_speed = np.random.uniform(speed_limit[0], speed_limit[1])
                attacker_x_speed = np.random.uniform(-attacker_speed, attacker_speed)
                attacker_y_speed = np.random.choice([-1,1]) * math.sqrt((attacker_speed ** 2)-(attacker_x_speed ** 2))
                attacker_spawn = intersecting_point - np.array([attacker_x_speed, attacker_y_speed]) * time2intersect
                if attacker_spawn[0] >= self.attacker_spawn_range[0] and attacker_spawn[0] <= self.attacker_spawn_range[1]\
                    and attacker_spawn[1] >= self.attacker_spawn_range[2] and attacker_spawn[1] <= self.attacker_spawn_range[3]:
                    in_range = True
            
            victim_spawn_transform = victim_spawn_location[i] + victim_spawn_z_rotation
            attacker_spawn_transform = attacker_spawn.tolist() + attacker_spawn_z_rotation
            subsettings = copy.deepcopy(settings)
            subsettings['victim']['spawn_transform'] = victim_spawn_transform
            subsettings['attacker']['spawn_transform'] = attacker_spawn_transform
            subsettings['attacker']['init_speed'] = [attacker_x_speed, attacker_y_speed, 0., attacker_speed]
            settings_list.append(subsettings)
            
        
        return settings_list