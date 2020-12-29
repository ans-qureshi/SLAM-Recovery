import argparse
import copy
import math
import numpy as np
import pygame
from pygame.locals import *
from timeit import default_timer as timer
import traceback
import os
from minos.lib import common
from minos.config.sim_args import parse_sim_args
from minos.lib.Simulator import Simulator
from minos.lib.util.ActionTraces import ActionTraces
from minos.lib.util.StateSet import StateSet
from minos.lib.util.VideoWriter import VideoWriter

import random,time

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


previous_action = 119
collision_counter = 10
final_time = 0
count=0

def get_angle(x_vector,y_vector):
    angle = 0.0
    if abs(x_vector) > 0:
        angle = math.atan2(y_vector, x_vector)*180/math.pi
        if angle >0:
            angle=angle-180
        else:
            angle=angle+180
        print('Direction to goal: ', angle)
    return angle


def get_random_action():
    actions = [119, 115, 97, 100, 276, 275]
    try:
        rand_no = random.randint(0, 5)
        next_action = actions[rand_no]
    except IndexError:
        next_action = actions[5]
    return next_action

def classifier():
    print("Running Classifier")
    os.system('/home/romi/SingleImageClassifier.py')
    fd = "/home/romi/abc2.txt"
    file = open(fd, 'r') 
    text = file.read() 
    if text=="Forward":
        return("119")
    elif text=="Back":
        return("115")
    elif text=="CW":
        return("275")
    elif text=="CCW":
        return("276")
    elif text=="Right":
        return("100")
    elif text=="Left":
        return("97")

normal_counter = 0
def generate_key_press(has_collided, direction, distance):
    global previous_action, collision_counter, normal_counter
    time.sleep(0.8)
    if normal_counter%2 == 0:
        next_action = 119
    else:
        angle = get_angle(x_vector=direction[2], y_vector=direction[0])
        #print('Angle: ', angle)

        if abs(angle) < 13:
            # The angle difference is very Small (Keep Moving Unless You Collide)
            next_action = 119
            if has_collided:
                print('Collision Detected: Taking Random Action')
                next_action = get_random_action()
            else:
                print('No Collision: Moving Forward')
                next_action = 119
        else:
            # Need to Adjust Angle
            print('Adjusting Angle')
            if angle > 0:
                # Turn Left
                next_action = 276

            else:
                # Turn Right
                next_action = 275

        if distance < 0.3:
            print("Goal Reached")
            time.sleep(3)
            scene_index = (scene_index + 1) % len(args.scene_ids)
            scene_id = args.scene_ids[scene_index]
            id = scene_dataset + '.' + scene_id
            print('next_scene loading %s ...' % id)
            sim.set_scene(id)
            sim.episode_info = sim.start()

    normal_counter = normal_counter + 1
    previous_action = next_action

    empty_keys = np.zeros(323, dtype='i')
    empty_keys[next_action] = 1
    return tuple(empty_keys)


def interactive_loop(sim, args):
    # initialize
    pygame.mixer.pre_init(frequency=8000, channels=1)
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
    global final_time
    global count
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
    has_collided=False
    direction=[0.0,0.0,0.0]
    distance=10
    total_frames=0
    pos = [-2.8103108374095114, 0.5275233, -17.989381395036254]

    ang=4.88692       #angle in radian
    tilt=0            #tilt angle in radian (keep it zero)
    print('\nMoving to Starting Point\t',pos,ang)
    sim.move_to(pos,ang,tilt)                     #define starting point here by pressing 'v'v
    '''
    House 17DRP5sb8fy
    Point A: 
    [1.3307827641472185, 0.53861988, -10.146044853235205]
    [3.5180930438444764, 0.566234, -1.8922092359314986]   dining room'
    
    [-1.4967893299263657, 0.5536211000000001, -9.28489025596529]

    Point B:
    [2.4621904181013585, 0.5086211, 1.905232439043702]
    [1.5371551074831127, 0.566234, -7.069940355796163]   bedroom

    [2.614761534032291, 0.5443598460000001, 2.082176819455741]

    House JeFG25nYj2p

    Point A:  
    [-6.74225409821948, 0.55074358, 9.639630625521972] lounge
    [-4.808781002856764, 0.579616, -3.2566622905687566]
   
    A: [6.161385541472788, 0.55074358, 1.0423787109815281]

    [-9.969136013947448, 0.5890924000000001, 4.443266425597109]
    [-3.8024725023701844, 0.5667730000000001, -3.897040174086085]  hallway

    Point B: (Set in env_config file)
    [5.324160089316311, 0.5293569, 1.0104915805896062]
    [0.09539571149637265, 0.579616, 8.280566401485888]  familyroom/lounge
   
    [3.234156347243452, 0.54744289, 5.132335702885289]   kitchen
    B: [-5.547085140683584, 0.55074358, -5.553243897709698]



    House ZMojNkEp431
    Point A:
    [-1.648939148401732, -0.316777, 21.77250735917539]
    [-7.257402680352993, -0.316777, 22.62233552621379]

    Point B:
    [1.5515085926612524, -0.316777, 27.586585491887465]

    House q9vSo1VnCiC
    Point A:
    [-9.295046451025712, 0.54650591, 9.32940161221944]

    Point B:
    [-2.93651621783526, 0.5278996, -8.961261587263957]
 
    House YVUC4YcDtcY (prob)
    Point A:
    [-22.444534161392085, 0.54291507, -16.989977211158624]

    Point B:
    [-30.20203545489092, 0.54291507, -7.346167078006564]

    House qoiz87JEwZ2
    Point A:
    [4.309296503924577, 0.69781, -2.2073896572614493]

    Point B:
    [12.669347833403256, -2.7127, -0.9483520109703518]
 
    '''
    while sim.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.running = False

        # read keys
        fd = "/home/romi/abc.txt"
        file = open(fd, 'r') 
        text = file.read() 
        text3="Tracking"
        text2=""
        text1="Track is lost"
        #print("Text :", text)
        #print("Count : " ,count) 
        time_taken = timer() - final_time 
        print(' fps=%f' % ( num_frames / time_taken))
        total_frames=total_frames+num_frames
        num_frames=0
        final_time=timer()
        if text==text3 and count >2:
            count=0
        elif text==text2 or text==text3 or count >=10:
            if count>=10:
                print('Failed Case, please recover tracks manually\n')   
                keys = pygame.key.get_pressed()      
            else:
                #keys = pygame.key.get_pressed()
                keys = generate_key_press(has_collided, direction,distance)
                #print('key pressed',action['name'])
                #time.sleep(0.5)
                open('/home/romi/abc2.txt', 'w').close()
                open('/home/romi/abc.txt', 'w').close()  

        elif text==text1:
            open('/home/romi/abc2.txt', 'w').close()
            open('/home/romi/abc.txt', 'w').close()
            print("\nTaking Random Steps to Recover\n")
            text4 = get_random_action()
            open('/home/romi/abc2.txt', 'w').close()
            open('/home/romi/abc.txt', 'w').close()
            empty_keys = np.zeros(323, dtype='i')
            empty_keys[text4] = 1
            keys= tuple(empty_keys)
            count+=1
            print("\nNumber of Random Movements done : ",count)

        
        
      #  keys = pygame.key.get_pressed()


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
        elif keys[K_v]:
            prev_key = 'v' if prev_key is not 'v' else ''
            if prev_key is 'v':
                pos=[0.37536343739988054, 0.49121938, 1.7367364232544902]
                ang=4.88692       #angle in radian
                tilt=0            #tilt angle in radian (keep it zero)
                print('\nMoving to Starting Point\t',pos,ang)
                sim.move_to(pos,ang,tilt)                     #define starting point here by pressing 'v'v
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
        direction = observation['measurements']['shortest_path_to_goal']['direction']
        distance = observation['measurements']['shortest_path_to_goal']['distance']
        print('Total Distance Remaining  ',distance)
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
            blit_img_to_surf(img, display_surf, config.get('position'), surf_key='map')

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
    print('time=%f sec, fps=%f' % (time_taken, total_frames / time_taken))
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
