import argparse
import copy
import math
import os
import numpy as np
import pygame
from pygame.locals import *
from timeit import default_timer as timer
import traceback
import json

from minos.lib import common
from minos.config.sim_args import parse_sim_args
from minos.lib.Simulator import Simulator
from minos.lib.util.ActionTraces import ActionTraces
from minos.lib.util.StateSet import StateSet
from minos.lib.util.VideoWriter import VideoWriter

import random
import time

from PIL import Image




REPLAY_MODES = ['actions', 'positions']
VIDEO_WRITER = None
TMP_SURFS = {}


def blit_img_to_surf(img, surf, position=(0, 0), surf_key='*'):
    global TMP_SURFS
    if len(img.shape) == 2:  # gray (y)
        img = np.dstack([img, img, img, np.ones(img.shape, dtype=np.uint8)*255])  # y -> yyy1
    else:
        img = img[:, :, [2, 1, 0, 3]]  # bgra -> rgba
    img_shape = (img.shape[0], img.shape[1])
    TMP_SURF = TMP_SURFS.get(surf_key)
    if not TMP_SURF or TMP_SURF.get_size() != img_shape:
        # print('create new surf %dx%d' % img_shape)
        TMP_SURF = pygame.Surface(img_shape, 0, 32)
        TMP_SURFS[surf_key] = TMP_SURF
    bv = TMP_SURF.get_view("0")
    bv.write(img.tostring())
    del bv
    surf.blit(TMP_SURF, position)


def display_episode_info(episode_info, display_surf, camera_outputs, show_goals=False):
    displayed = episode_info.get('displayed',0)
    if displayed < 1:
        print('episode_info', {k: episode_info[k] for k in episode_info if k != 'goalObservations'})
        if show_goals and 'goalObservations' in episode_info:
            # NOTE: There can be multiple goals with separate goal observations for each
            # We currently just handle one
            goalObservations = episode_info['goalObservations']
            if len(goalObservations) > 0:
                # Call display_response but not write to video
                display_response(goalObservations[0], display_surf, camera_outputs, print_observation=False, write_video=False)
        episode_info['displayed'] = displayed + 1


def draw_forces(forces, display_surf, area):
    r = 5
    size = round(0.45*min(area.width, area.height)-r)
    center = area.center
    pygame.draw.rect(display_surf, (0,0,0), area, 0)  # fill with black
    # assume forces are radially positioned evenly around agent
    # TODO: Actual get force sensor positions and visualize them
    dt = -2*math.pi/forces.shape[0]
    theta = math.pi/2
    for i in range(forces.shape[0]):
        x = round(center[0] + math.cos(theta)*size)
        y = round(center[1] + math.sin(theta)*size)
        width = 0 if forces[i] else 1
        pygame.draw.circle(display_surf, (255,255,0), (x,y), r, width)
        theta += dt


def draw_offset(offset, display_surf, area, color=(0,0,255)):
    dir = (offset[0], offset[2])
    mag = math.sqrt(dir[0]*dir[0] + dir[1]*dir[1])
    if mag:
        dir = (dir[0]/mag, dir[1]/mag)
    size = round(0.45*min(area.width, area.height))
    center = area.center
    target = (round(center[0]+dir[0]*size), round(center[1]+dir[1]*size))
    pygame.draw.rect(display_surf, (0,0,0), area, 0)  # fill with black
    pygame.draw.circle(display_surf, (255,255,255), center, size, 0)
    pygame.draw.line(display_surf, color, center, target, 1)
    pygame.draw.circle(display_surf, color, target, 4, 0)


def display_response(response, display_surf, camera_outputs, print_observation=False, write_video=False):
    global VIDEO_WRITER
    observation = response.get('observation')
    sensor_data = observation.get('sensors')
    measurements = observation.get('measurements')

    def printable(x): return type(x) is not bytearray and type(x) is not np.ndarray
    if observation is not None and print_observation:
        simple_observations = {k: v for k, v in observation.items() if k not in ['measurements', 'sensors']}
        dicts = [simple_observations, observation.get('measurements'), observation.get('sensors')]
        for d in dicts:
            for k, v in d.items():
                if type(v) is not dict:
                    info = '%s: %s' % (k,v)
                    print(info[:75] + (info[75:] and '..'))
                else:
                    print('%s: %s' % (k, str({i: v[i] for i in v if printable(v[i])})))
        if 'forces' in sensor_data:
            print('forces: %s' % str(sensor_data['forces']['data']))
        if 'info' in response:
            print('info: %s' % str(response['info']))

    if 'offset' in camera_outputs:
        draw_offset(measurements.get('offset_to_goal'), display_surf, camera_outputs['offset']['area'])

    for obs, config in camera_outputs.items():
        if obs not in sensor_data:
            continue
        if obs == 'forces':
            draw_forces(sensor_data['forces']['data'], display_surf, config['area'])
            continue
        img = sensor_data[obs]['data']
        img_viz = sensor_data[obs].get('data_viz')
        if obs == 'depth':
            img *= (255.0 / img.max())  # naive rescaling for visualization
            img = img.astype(np.uint8)
        elif img_viz is not None:
            img = img_viz
        blit_img_to_surf(img, display_surf, config.get('position'))

        # TODO: consider support for writing to video of all camera modalities together
        if obs == 'color':
            if write_video and VIDEO_WRITER:
                if len(img.shape) == 2:
                    VIDEO_WRITER.add_frame(np.dstack([img, img, img]))  # yyy
                else:
                    VIDEO_WRITER.add_frame(img[:, :, :-1])  # rgb

    if 'audio' in sensor_data:
        audio_data = sensor_data['audio']['data']
        pygame.sndarray.make_sound(audio_data).play()
        # pygame.mixer.Sound(audio_data).play()


def write_text(display_surf, text, position, font=None, fontname='monospace', fontsize=12, color=(255,255,224), align=None):
    """
    text -> string of text.
    fontname-> string having the name of the font.
    fontsize -> int, size of the font.
    color -> tuple, adhering to the color format in pygame.
    position -> tuple (x,y) coordinate of text object.
    """

    font_object = font if font is not None else pygame.font.SysFont(fontname, fontsize)
    text_surface = font_object.render(text, True, color)
    if align is not None:
        text_rectangle = text_surface.get_rect()
        if align == 'center':
            text_rectangle.center = position[0], position[1]
        else:
            text_rectangle.topleft = position
        display_surf.blit(text_surface, text_rectangle)
    else:
        display_surf.blit(text_surface, position)


collision_counter = 10


def get_angle(x_vector,y_vector):
    angle = 0.0
    if abs(x_vector) > 0:
        angle = math.atan2(y_vector, x_vector)*180/math.pi
        if angle >0:
            angle=angle-180
        else:
            angle=angle+180
        print('Atan2 Angle: ', angle)
    return angle


def get_random_action():
    actions = [119, 115, 97, 100, 276, 275]
    try:
        rand_no = random.randint(0, 5)
        next_action = actions[rand_no]
    except IndexError:
        next_action = actions[5]
    return next_action

previous_action = 0
next_action = 0
scan_direction = 'right'
move_direction = 'forward'
exploration_counter = 0
image_counter = 0
explore = False
saved_previous_action = 0
save_image = False

def start_explore(observation, args):
    global exploration_counter, next_action, saved_previous_action, previous_action, explore, image_counter, save_image
  
    if exploration_counter == 0:
        # current_image.save(f"{image_counter:8f}")
        save_image = True
        next_action = 275  # CW
        save_dataset_image(observation, next_action, args)
        saved_previous_action = previous_action
        explore = True
        save_image = True
        image_counter = image_counter + 1
    elif exploration_counter == 1:
        next_action = 276  # CCW
        explore = True
    elif exploration_counter == 2:
        next_action = 276  # CCW
        explore = True
        save_image = True
        image_counter = image_counter + 1
    elif exploration_counter == 3:
        next_action = 275  # CW
        explore = True
    elif exploration_counter == 4:
        next_action = 97    #left
        explore = True
        save_image = True
        image_counter = image_counter + 1
    elif exploration_counter == 5:
        next_action = 100   #right
        explore = True
    elif exploration_counter == 6:
        next_action = 100   #right
        explore = True
        save_image = True
        image_counter = image_counter + 1
    elif exploration_counter == 7:
        next_action = 97    #left
        explore = True
    elif exploration_counter == 8:
        next_action = 119   #forward
        explore = True
        save_image = True
        image_counter = image_counter + 1
    elif exploration_counter == 9:
        next_action = 115    #back
        explore = True
    elif exploration_counter == 10:
        next_action = 115
        explore = True
        save_image = True
        image_counter = image_counter + 1
    elif exploration_counter == 11:
        next_action = 119
        explore = False
        save_image = False
        image_counter = image_counter + 1
    exploration_counter = exploration_counter + 1


def generate_key_press_exploration(has_collided,observation, args,sim):
    global previous_action, collision_counter, scan_direction, exploration_counter, image_counter, next_action, explore, saved_previous_action

    #time.sleep(1)
    scene_index = 0
    scene_index = (scene_index + 1) % len(args.scene_ids)
    scene_id = args.scene_ids[scene_index]
    scene_dataset = args.scene.dataset
    path = "/home/romi/frames2/"+scene_id+"/annotations.json"

    # Tracking......
    if previous_action == 0:  # Start of Simulation
        next_action = 100  # Start with Right
        explore = True
    else:  # Ongoing Simulation (Not the Start)
        if has_collided is not True:  # If you've not collided
            if explore is True:
                start_explore(observation, args)
            else:
                previous_action = saved_previous_action
                if previous_action != 119 and previous_action != 115:
                    # If last action was not forward and neither backward, keep moving in the same direction
                    if (previous_action == 97 and scan_direction == 'left') or (previous_action == 100 and scan_direction == 'right'):
                        next_action = previous_action
                    else:
                        if move_direction == 'forward':
                            next_action = 119  # Move Forward for next scan
                        elif move_direction == 'backward':
                            next_action = 115  # Move Backward for next scan
                else:  # If You've not collided and Last Action was Forward or Backward, Reverse the direction of Scan
                    if scan_direction == 'right':
                        scan_direction = 'left'
                        next_action = 97
                    else:
                        scan_direction = 'right'
                        next_action = 100
                explore = True
                exploration_counter = 0
        else:  # Collision Occurred while moving forward or backward for new scan
            # Take an opposite step from the direction of scan
            exploration_counter = 0
            if 'state_set' in args :
                state = args.state_set.get_next_state()
                if not state:  # roll over to beginning
                    print('Restarting from beginning of states file...')
                    state = args.state_set.get_next_state()
                id = scene_dataset + '.' + state['scene_id']
                print('next_scene loading %s ...' % id)
                sim.set_scene(id)
                sim.move_to(state['start']['position'], state['start']['angle'])
                sim.episode_info = sim.start()
                with open(path, 'w') as json_file:
                    json.dump(annotation_arr, json_file)
            else:
                scene_index = (scene_index + 1) % len(args.scene_ids)
                scene_id = args.scene_ids[scene_index]
                id = scene_dataset + '.' + scene_id
                print('next_scene loading %s ...' % id)
                sim.set_scene(id)
                sim.episode_info = sim.start()
                with open(path, 'w') as json_file:
                    json.dump(annotation_arr, json_file)

            print('------------------May be Stuck Here----------------------')
    print(previous_action)
    previous_action = next_action

    empty_keys = np.zeros(323, dtype='i')
    empty_keys[next_action] = 1
    return tuple(empty_keys)


def get_observed_image(observation):
    current_image_shape = observation['sensors']['color']['data'].shape
    current_image_array = np.reshape(observation['sensors']['color']['data'], current_image_shape)
    current_image = Image.frombytes('RGBA', (current_image_shape[0], current_image_shape[1]), current_image_array)
    return current_image


def get_merged_image(first_image, second_image):
    start_image_gray = first_image.convert(mode='L')
    end_image_gray = second_image.convert(mode='L')
    dummy_channel = Image.new(mode='L', size=end_image_gray.size)
    merged_image = Image.merge(mode='RGB', bands=(start_image_gray, end_image_gray, dummy_channel))
    return merged_image


visited_pos = list()


def is_close(current_pos):
    global visited_pos
    threshold_x = 0.01
    threshold_y = 0.01
    threshold_angle = 2

    for v in visited_pos:
        if abs(current_pos[0]-v[0]) < threshold_x and abs(current_pos[1] - v[1]) < threshold_y and abs(current_pos[2] - v[2]) < threshold_angle:
            return True
    return False


annotation_dict={}
annotation_arr = []
base_image = ""
dict_appended =False
cc = 0

def save_dataset_image(observation, next_action, args):
    global annotation_dict, annotation_arr, base_image, dict_appended, save_image, cc
    scene_index = 0

    scene_index = (scene_index + 1) % len(args.scene_ids)
    scene_id = args.scene_ids[scene_index]
    id = scene_id
    current_image = get_observed_image(observation)
    path = "/home/romi/frames2/"+id+"/"
    if not os.path.exists(path):
        os.makedirs(path)
    if save_image:
        current_image.save(path + "{:08d}".format(image_counter)+"_"+str(next_action)+".png")
    if exploration_counter == 0 and cc == 0 and next_action == 275:
        cc = 1
        if image_counter != 0 and dict_appended == False:
            annotation_arr.append(annotation_dict)
            dict_appended = True
        print("Annotation Array: ", annotation_arr)
        annotation_dict = {}
        base_image = "{:08d}".format(image_counter)+"_"+str(next_action)+".png"
        annotation_dict[base_image] = {}
    if next_action == 275 and save_image is True and cc == 1:
        cc = 0
        annotation_dict[base_image]['cw'] = "{:08d}".format(image_counter+1)+"_"+str(next_action)+".png"
        dict_appended = False
    elif next_action== 276 and save_image is True:
        annotation_dict[base_image]['ccw'] = "{:08d}".format(image_counter)+"_"+str(next_action) + ".png"
    elif next_action == 97 and save_image is True:
        annotation_dict[base_image]['left'] = "{:08d}".format(image_counter)+"_"+str(next_action)+".png"
    elif next_action == 100 and save_image is True:
        annotation_dict[base_image]['right'] = "{:08d}".format(image_counter)+"_"+str(next_action)+".png"
    elif next_action == 119 and save_image is True:
        annotation_dict[base_image]['forward'] = "{:08d}".format(image_counter)+"_"+str(next_action)+".png"
    elif next_action == 115 and save_image is True:
        annotation_dict[base_image]['backward'] = "{:08d}".format(image_counter)+"_"+str(next_action)+".png"
    save_image = False


def interactive_loop(sim, args):
    global visited_pos
    # initialize
    pygame.mixer.pre_init(frequency = 8000, channels = 1)
    pygame.init()
    pygame.key.set_repeat(500, 50)  # delay, interval
    clock = pygame.time.Clock()
    # Set up display
    font_spacing = 20
    display_height = args.height + font_spacing*3
    all_camera_observations = ['color', 'depth', 'normal', 'objectId', 'objectType', 'roomId', 'roomType']
    label_positions = {
        'curr': {},
        'goal': {}
    }
    camera_outputs = {
        'curr': {},
        'goal': {}
    }

    # row with observations and goals
    nimages = 0
    for obs in all_camera_observations:
        if args.observations.get(obs):
            label_positions['curr'][obs] = (args.width*nimages, font_spacing*2)
            camera_outputs['curr'][obs] = { 'position': (args.width*nimages, font_spacing*3) }
            if args.show_goals:
                label_positions['goal'][obs] = (args.width*nimages, display_height + font_spacing*2)
                camera_outputs['goal'][obs] = { 'position': (args.width*nimages, display_height + font_spacing*3) }
            nimages += 1


    if args.show_goals:
        display_height += args.height + font_spacing*3

    # Row with offset and map
    plot_size = max(min(args.height, 128), 64)
    display_height += font_spacing
    label_positions['curr']['offset'] = (0, display_height)
    camera_outputs['curr']['offset'] = { 'area': pygame.Rect(0, display_height + font_spacing, plot_size, plot_size)}

    next_start_x = plot_size
    if args.observations.get('forces'):
        label_positions['curr']['forces'] = (next_start_x, display_height)
        camera_outputs['curr']['forces'] = { 'area': pygame.Rect(next_start_x, display_height + font_spacing, plot_size, plot_size)}
        next_start_x += plot_size

    if args.observations.get('map'):
        label_positions['map'] = (next_start_x, display_height)
        camera_outputs['map'] = { 'position': (next_start_x, display_height + font_spacing) }

    display_height += font_spacing
    display_height += plot_size

    display_shape = [max(args.width * nimages, next_start_x), display_height]
    display_surf = pygame.display.set_mode((display_shape[0], display_shape[1]), pygame.RESIZABLE | pygame.DOUBLEBUF)

    # Write text
    label_positions['title'] = (display_shape[0]/2, font_spacing/2)
    write_text(display_surf, 'MINOS', fontsize=20, position = label_positions['title'], align='center')
    write_text(display_surf, 'dir_to_goal', position = label_positions['curr']['offset'])
    if args.observations.get('forces'):
        write_text(display_surf, 'forces', position = label_positions['curr']['forces'])
    if args.observations.get('map'):
        write_text(display_surf, 'map', position = label_positions['map'])
    write_text(display_surf, 'observations | controls: WASD+Arrows', position = (0, font_spacing))
    if args.show_goals:
        write_text(display_surf, 'goal', position = (0, args.height + font_spacing*3 + font_spacing))
    for obs in all_camera_observations:
        if args.observations.get(obs):
            write_text(display_surf, obs, position = label_positions['curr'][obs])
            if args.show_goals:
                write_text(display_surf, obs, position = label_positions['goal'][obs])

    # Other initialization
    scene_index = 0
    scene_dataset = args.scene.dataset

    init_time = timer()
    num_frames = 0
    prev_key = ''
    replay = args.replay
    action_traces = args.action_traces
    action_trace = action_traces.curr_trace() if action_traces is not None else None
    replay_auto = False
    replay_mode = args.replay_mode
    replay_mode_index = REPLAY_MODES.index(replay_mode)
    print('***\n***')
    print('CONTROLS: WASD+Arrows = move agent, R = respawn, N = next state/scene, O = print observation, Q = quit')
    if replay:
        print('P = toggle auto replay, E = toggle replay using %s '
              % str([m + ('*' if m == replay_mode else '') for m in REPLAY_MODES]))
    print('***\n***')
    has_collided = False
    direction = [0.0, 0.0, 0.0]
    prev_image = None
    current_image = None
    generate_dataset = False

    i = 0

    observation = None

    while sim.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.running = False

        keys = generate_key_press_exploration(has_collided, observation, args,sim)

        if keys[0] == 1:  # We are stuck!
            break

        #keys = pygame.key.get_pressed()

        print_next_observation = False
        if keys[K_q]:
            break

        if keys[K_o]:
            print_next_observation = True
        elif keys[K_n]:
            prev_key = 'n' if prev_key is not 'n' else ''
            if 'state_set' in args and prev_key is 'n':
                state = args.state_set.get_next_state()
                if not state:  # roll over to beginning
                    print('Restarting from beginning of states file...')
                    state = args.state_set.get_next_state()
                id = scene_dataset + '.' + state['scene_id']
                print('next_scene loading %s ...' % id)
                sim.set_scene(id)
                sim.move_to(state['start']['position'], state['start']['angle'])
                sim.episode_info = sim.start()
            elif prev_key is 'n':
                scene_index = (scene_index + 1) % len(args.scene_ids)
                scene_id = args.scene_ids[scene_index]
                id = scene_dataset + '.' + scene_id
                print('next_scene loading %s ...' % id)
                sim.set_scene(id)
                sim.episode_info = sim.start()
        elif keys[K_r]:
            prev_key = 'r' if prev_key is not 'r' else ''
            if prev_key is 'r':
                sim.episode_info = sim.reset()
        else:
            # Figure out action
            action = {'name': 'idle', 'strength': 1, 'angle': math.radians(5)}
            actions = []
            if replay:
                unprocessed_keypressed = any(keys)
                if keys[K_p]:
                    prev_key = 'p' if prev_key is not 'p' else ''
                    if prev_key == 'p':
                        replay_auto = not replay_auto
                        unprocessed_keypressed = False
                elif keys[K_e]:
                    prev_key = 'e' if prev_key is not 'e' else ''
                    if prev_key == 'e':
                        replay_mode_index = (replay_mode_index + 1) % len(REPLAY_MODES)
                        replay_mode = REPLAY_MODES[replay_mode_index]
                        unprocessed_keypressed = False
                        print('Replay using %s' % replay_mode)

                if replay_auto or unprocessed_keypressed:
                    # get next action and do it
                    rec = action_trace.next_action_record()
                    if rec is None:
                        # go to next trace
                        action_trace = action_traces.next_trace()
                        start_state = action_trace.start_state()
                        print('start_state', start_state)
                        sim.configure(start_state)
                        sim.episode_info = sim.start()
                    else:
                        if replay_mode == 'actions':
                            actnames = rec['actions'].split('+')
                            for actname in actnames:
                                if actname != 'reset':
                                    act = copy.copy(action)
                                    act['name'] = actname
                                    actions.append(act)
                        elif replay_mode == 'positions':
                            sim.move_to([rec['px'], rec['py'], rec['pz']], rec['rotation'])
            else:
                if keys[K_w]:
                    action['name'] = 'forwards'
                    print('Action: Forward')
                elif keys[K_s]:
                    action['name'] = 'backwards'
                    print('Action: Backward')
                elif keys[K_LEFT]:
                    # ASCII Code 276
                    action['name'] = 'turnLeft'
                    print('Action: Rotate Left')
                elif keys[K_RIGHT]:
                    # ASCII Code 275
                    action['name'] = 'turnRight'
                    print('Action: Rotate Right')
                elif keys[K_a]:
                    action['name'] = 'strafeLeft'
                    print('Action: Strafe Left')
                elif keys[K_d]:
                    action['name'] = 'strafeRight'
                    print('Action: Strafe Right')
                elif keys[K_UP]:
                    action['name'] = 'lookUp'
                elif keys[K_DOWN]:
                    action['name'] = 'lookDown'
                else:
                    action['name'] = 'idle'
                actions = [action]

        # step simulator and get observation
        response = sim.step(actions, 1)
        if response is None:
            break

        display_episode_info(sim.episode_info, display_surf, camera_outputs['goal'], show_goals=args.show_goals)

        # Handle map
        observation = response.get('observation')

        # Save Image if Needed
        save_dataset_image(observation, next_action, args)

        distance_from_origin = math.sqrt((response['info']['agent_state']['position'][0])**2 + (response['info']['agent_state']['position'][2])**2)
        print('Position: ', response['info']['agent_state']['position'][0], response['info']['agent_state']['position'][1], response['info']['agent_state']['position'][2])
        print('Distance from Origin: ', distance_from_origin)
        orientation = response['info']['agent_state']['position']
        orientation_angle = get_angle(orientation[2], orientation[0])
        print('Orientation Angle: ', orientation_angle)
        position_orientation = (response['info']['agent_state']['position'][0], response['info']['agent_state']['position'][2], orientation_angle)
        # print('Visited POS: ', visited_pos, type(visited_pos))
        if not is_close(position_orientation):
            visited_pos.append(position_orientation)
            explore = 1

        if generate_dataset:
            if current_image is not None:
                prev_image = current_image
            current_image = get_observed_image(observation)

            if prev_image is not None and has_collided is False:
                # Only Save Image if there is no Collision and There exists a Previous Observation
                print('Save the Merged Image')
                merged_image = get_merged_image(first_image=prev_image, second_image=current_image)
                merged_image.save(fp=os.path.join('/home/sohail/' + action['name'] + str(i) + '.jpg'))
                i = i+1

        '''if observation['measurements']['shortest_path_to_goal']['distance'] < 0.2:
            print('Goal Reached')
            direction = [0.0, 0.0, 0.0]
            sim.running = False
        else:'''
        #direction = observation['measurements']['shortest_path_to_goal']['direction']
        # print(get_angle(direction[2], direction[0]))
        has_collided = observation['collision']
        map = observation.get('map')
        if map is not None:
            # TODO: handle multiple maps
            if isinstance(map, list):
                map = map[0]
            config = camera_outputs['map']
            img = map['data']
            rw = map['shape'][0] + config.get('position')[0]
            rh = map['shape'][1] + config.get('position')[1]
            w = display_surf.get_width()
            h = display_surf.get_height()
            if w < rw or h < rh:
                # Resize display (copying old stuff over)
                old_display_surf = display_surf.convert()
                display_surf = pygame.display.set_mode((max(rw,w), max(rh,h)), pygame.RESIZABLE | pygame.DOUBLEBUF)
                display_surf.blit(old_display_surf, (0,0))
                write_text(display_surf, 'map', position = label_positions['map'])
            blit_img_to_surf(img, display_surf, config.get('position'), surf_key='map')  # Map is drawn here

        # Handle other response
        display_response(response, display_surf, camera_outputs['curr'], print_observation=print_next_observation, write_video=True)
        pygame.display.flip()
        num_frames += 1
        clock.tick(30)  # constraint to max 30 fps

    # NOTE: log_action_trace handled by javascript side
    # if args.log_action_trace:
    #     trace = sim.get_action_trace()
    #     print(trace['data'])

    # cleanup and quit
    time_taken = timer() - init_time
    print('time=%f sec, fps=%f' % (time_taken, num_frames / time_taken))
    print('Thank you for playing - Goodbye!')
    pygame.quit()


def main():
    global VIDEO_WRITER
    parser = argparse.ArgumentParser(description='Interactive interface to Simulator')
    parser.add_argument('--navmap', action='store_true',
                        default=False,
                        help='Use navigation map')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--state_set_file',
                       help='State set file')
    group.add_argument('--replay',
                       help='Load and replay action trace from file')
    group.add_argument('--replay_mode',
                       choices=REPLAY_MODES,
                       default='positions',
                       help='Use actions or positions for replay')
    group.add_argument('--show_goals', action='store_true',
                       default=False,
                       help='show goal observations')

    args = parse_sim_args(parser)
    args.visualize_sensors = True
    sim = Simulator(vars(args))
    common.attach_exit_handler(sim)

    if 'state_set_file' in args and args.state_set_file is not None:
        args.state_set = StateSet(args.state_set_file, 1)
    if 'save_video' in args and len(args.save_video):
        filename = args.save_video if type(args.save_video) is str else 'out.mp4'
        is_rgb = args.color_encoding == 'rgba'
        VIDEO_WRITER = VideoWriter(filename, framerate=24, resolution=(args.width, args.height), rgb=is_rgb)
    if 'replay' in args and args.replay is not None:
        print('Initializing simulator using action traces %s...' % args.replay)
        args.action_traces = ActionTraces(args.replay)
        action_trace = args.action_traces.next_trace()
        sim.init()
        start_state = action_trace.start_state()
        print('start_state', start_state)
        sim.configure(start_state)
    else:
        args.action_traces = None
        args.replay = None

    try:
        print('Starting simulator...')
        ep_info = sim.start()
        if ep_info:
            print('observation_space', sim.get_observation_space())
            sim.episode_info = ep_info
            print('Simulator started.')
            interactive_loop(sim, args)
    except:
        traceback.print_exc()
        print('Error running simulator. Aborting.')

    if sim is not None:
        sim.kill()
        del sim

    if VIDEO_WRITER is not None:
        VIDEO_WRITER.close()


if __name__ == "__main__":
    main()
