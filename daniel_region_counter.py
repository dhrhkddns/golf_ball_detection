from collections import defaultdict
from pathlib import Path


import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

import threading
import pygame
import time

track_history = defaultdict(list)                    
                                                                                       

def golf_sound(): #홀 안에 들어갔을때
    pygame.mixer.init()
    pygame.mixer.music.load('C:\\Users\\omyra\\OneDrive\\바탕 화면\\Hey\\ultralytics\\examples\\YOLOv8-Region-Counter\\minigolf-putt-right-into-the-hole.mp3') # 소리 파일 경로
    pygame.mixer.music.play()

def golf_out_sound(): #골프공이 그린존 바깥으로 벗어났을 때
    pygame.mixer.init()
    pygame.mixer.music.load('C:\\Users\\omyra\\OneDrive\\바탕 화면\\Hey\\ultralytics\\examples\\YOLOv8-Region-Counter\\jazzy_fail.wav') # 소리 파일 경로
    pygame.mixer.music.play()

def start_sound(): #골프공을 선에 맞춰주세요
    pygame.mixer.init()
    pygame.mixer.music.load('C:\\Users\\omyra\\OneDrive\\바탕 화면\\Hey\\ultralytics\\examples\\YOLOv8-Region-Counter\\골프공을선에맞춰주세요.mp3') # 소리 파일 경로
    pygame.mixer.music.play()   

def ready_putting(): #준비됐으면 퍼팅! 사운드
    pygame.mixer.init()
    pygame.mixer.music.load('C:\\Users\\omyra\\OneDrive\\바탕 화면\\Hey\\ultralytics\\examples\\YOLOv8-Region-Counter\\준비됐으면퍼팅.wav') # 소리 파일 경로
    pygame.mixer.music.play()     

def golf_shot(): #준비됐으면 퍼팅! 사운드
    pygame.mixer.init()
    pygame.mixer.music.load('C:\\Users\\omyra\\OneDrive\\바탕 화면\\Hey\\ultralytics\\examples\\YOLOv8-Region-Counter\\golf_shot.wav') # 소리 파일 경로
    pygame.mixer.music.play()    
# def music_function():
#     pygame.mixer.music.play()


#     # 재생 종료까지 대기 #이거 없이 play()만 하면 안돼...
#     while pygame.mixer.music.get_busy():
#         continue
# my_thread = threading.Thread(target=music_function) #스레드, 실행해놓고 다음껄로 해서 프레임 계속~~


counting_regions = [
    {
         'name': 'small_hall',
         'polygon': Polygon([(75.0, 325.0), (74.92314112161293, 324.2196387119355), (74.69551813004514, 323.46926627053966), (74.32587844921018, 322.77771906792157), (73.82842712474618, 322.1715728752538), (73.2222809320784, 321.67412155078983), (72.53073372946037, 321.3044818699548), (71.78036128806451, 321.0768588783871), (71.0, 321.0), (70.21963871193549, 321.0768588783871), (69.46926627053963, 321.3044818699548), (68.7777190679216, 321.67412155078983), (68.17157287525382, 322.1715728752538), (67.67412155078982, 322.77771906792157), (67.30448186995486, 323.46926627053966), (67.07685887838707, 324.2196387119355), (67.0, 325.0), (67.07685887838707, 325.7803612880645), (67.30448186995486, 326.53073372946034), (67.67412155078982, 327.22228093207843), (68.17157287525382, 327.8284271247462), (68.7777190679216, 328.32587844921017), (69.46926627053965, 328.6955181300452), (70.21963871193549, 328.9231411216129), (71.0, 329.0), (71.78036128806451, 328.9231411216129), (72.53073372946037, 328.6955181300452), (73.2222809320784, 328.32587844921017), (73.82842712474618, 327.8284271247462), (74.32587844921018, 327.22228093207843), (74.69551813004514, 326.53073372946034), (74.92314112161291, 325.7803612880645)]),  # Polygon points
         'counts': 0,
         'dragging': False,
         'region_color': (6, 97, 255),  # BGR Value
         'text_color': (255, 255, 255)  # Region Text Color
     },
    {
        'name': 'big_hall',
        'polygon': Polygon([(71.0, 362.0), (70.86549696282262, 360.6343677458871), (70.46715672757901, 359.3212159734444), (69.82028728611782, 358.11100836886277), (68.94974746830583, 357.0502525316942), (67.88899163113722, 356.17971271388217), (66.67878402655563, 355.532843272421), (65.3656322541129, 355.1345030371774), (64.0, 355.0), (62.634367745887104, 355.1345030371774), (61.321215973444374, 355.532843272421), (60.111008368862784, 356.17971271388217), (59.05025253169417, 357.0502525316942), (58.17971271388218, 358.11100836886277), (57.53284327242099, 359.3212159734444), (57.13450303717739, 360.6343677458871), (57.0, 362.0), (57.13450303717739, 363.3656322541129), (57.53284327242099, 364.6787840265556), (58.17971271388218, 365.88899163113723), (59.05025253169417, 366.9497474683058), (60.11100836886278, 367.82028728611783), (61.321215973444374, 368.467156727579), (62.6343677458871, 368.8654969628226), (64.0, 369.0), (65.3656322541129, 368.8654969628226), (66.67878402655563, 368.467156727579), (67.88899163113722, 367.82028728611783), (68.94974746830583, 366.9497474683058), (69.82028728611782, 365.88899163113723), (70.46715672757901, 364.6787840265556), (70.86549696282262, 363.3656322541129)]),  # Polygon points
        'counts': 0,
        'dragging': False,
        'region_color': (37, 255, 225),  # BGR Value
        'text_color': (0, 0, 0),  # Region Text Color
    },
    {
         'name': 'green_hall',
         'polygon': Polygon([(43, 386), (45, 323), (51, 307), (62, 304), (638, 310), (638, 384)]),  # Polygon points
         'counts': 0,
         'dragging': False,
         'region_color': (255, 42, 4),  # BGR Value 파란색
         'text_color': (255, 255, 255)  # Region Text Color
    },
    {
         'name': 'start_region',
         'polygon': Polygon([(517, 308), (541, 308), (541, 384), (517, 384)]),  # Polygon points
         'counts': 0,
         'dragging': False,
         'region_color': (0, 0, 0),  # BGR Value 검은색
         'text_color': (255, 255, 255),  # Region Text Color
         'in' : False
    },
    {
         'name': 'start_line', #시작선 여기 넘어가면 치는걸로 간주됨
         'polygon': Polygon([(518, 309), (518, 384),(520, 384), (520, 309)]),  # Polygon points
         'counts': 0,
         'dragging': False, 
         'region_color': (255, 255, 255),  # BGR Value 흰색
         'text_color': (255, 255, 255)  # Region Text Color
    }    ]
current_region = counting_regions
#[(517, 308), (541, 309), (541, 384), (517, 382)] 초기 골프공 치는 영역
#[(43, 386), (45, 323), (51, 307), (62, 304), (638, 310), (638, 384)]
japyo_lst = []

def mouse_callback(event, x, y, flags, param):
    """Mouse call back event."""
    global current_region

    # Mouse left button down event    
    if event == cv2.EVENT_LBUTTONDOWN:
        # japyo_lst.append((x,y))
        for region in counting_regions:
            if region['polygon'].contains(Point((x, y))):
                current_region = region
                current_region['dragging'] = True
                current_region['offset_x'] = x
                current_region['offset_y'] = y
                #japyo_lst.append([x,y])

    # Mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region['dragging']:
            dx = x - current_region['offset_x']
            dy = y - current_region['offset_y']
            current_region['polygon'] = Polygon([
                (p[0] + dx, p[1] + dy) for p in current_region['polygon'].exterior.coords])
            current_region['offset_x'] = x
            current_region['offset_y'] = y
            # print(list(current_region['polygon'].exterior.coords)[0][0])

   # Mouse left button up event
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region['dragging']:
            current_region['dragging'] = False


def run(
    weights='C:\\Users\\omyra\\OneDrive\\바탕 화면\\Hey\\small_golf_best_openvino_model',
    source=1, #웹캠 0 1 2
    device='cpu',
    view_img=True,
    save_img=True,
    exist_ok=False,
    classes=None,
    line_thickness=0, #count 글자
    track_thickness=2, #track했을때 볼 수 있음
    region_thickness=1,#경계선 변 두께
):
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        classes (list): classes to detect and track
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """
    vid_frame_count = 0
    first_out_sound = 0
    last_golf_out_sound_played = 0

    first_line_second = 0
    last_line_sound_count = 0

    golf_in_count = 0
    golf_in_second = 0
    
    golf_shot_sound = 0



    # Setup Model
    model = YOLO(f'{weights}')
    #model.to('cuda') if device == '0' else model.to('cpu')

    # Extract classes names
    #names = model.model.names
    print("--------------------------")
    #print(names) {0: 'golf'}


    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    print(frame_width)
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    print(str(frame_width) + " " + str(frame_height))
    # Output setup
    save_dir = increment_path(Path('ultralytics_rc_output') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / 'result.mp4'), fourcc, fps, (frame_width, frame_height))

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        
        if not success:
            break
        vid_frame_count += 1 #진짜 진행되는 프레임!!
        current_time = time.time()
        # print(current_time)
        if vid_frame_count == 10:
            threading.Thread(target=start_sound).start()  # 새로운 쓰레드에서 소리 재생
        
        # Extract the results
        results = model.track(frame, persist=True, classes=classes)
                                                                                         
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()
            #print(clss)
            #print(type(clss))
            annotator = Annotator(frame, line_width=line_thickness, example=str({0: 'golf'}))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str("GOLF_BALL"), color=colors(0, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center
                print(bbox_center)
                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 5: 
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)
                          

            #작은 홀 안에 들어갔을 때
            if current_region[0]['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                current_region[0]['counts'] += 1
                
                if golf_in_count == 0:
                    threading.Thread(target=golf_sound).start()  # 새로운 쓰레드에서 소리 재생
                    golf_in_count += 1
                    golf_in_second = current_time

                if current_time - golf_in_second > 5 and golf_in_count != 0:
                    threading.Thread(target=golf_sound).start()  # 새로운 쓰레드에서 소리 재생
                    golf_in_count += 1
                    golf_in_second = current_time    
            
            #큰 홀 안에 들어갔을 때
            if current_region[1]['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                current_region[1]['counts'] += 1
                
                if golf_in_count == 0:
                    threading.Thread(target=golf_sound).start()  # 새로운 쓰레드에서 소리 재생
                    golf_in_count += 1
                    golf_in_second = current_time

                if current_time - golf_in_second > 5 and golf_in_count != 0:
                    threading.Thread(target=golf_sound).start()  # 새로운 쓰레드에서 소리 재생
                    golf_in_count += 1
                    golf_in_second = current_time    
            
            #그린 존 바깥으로 공이 나갈때
            if current_region[2]['polygon'].contains(Point((bbox_center[0], bbox_center[1]))) != True:                  
                current_region[2]['counts'] += 1
                

                if first_out_sound == 0:
                    threading.Thread(target=golf_out_sound).start()
                    first_out_sound += 1
                    last_golf_out_sound_played = current_time

                if current_time - last_golf_out_sound_played > 5 and first_out_sound != 0:  # 마지막으로 소리를 재생한 후 5초 이상 지났다면
                    threading.Thread(target=golf_out_sound).start()  # 새로운 쓰레드에서 소리 재생
                    last_golf_out_sound_played = current_time  # 재생 시간 업데이트       
            
            #초기 라인에 공을 놔뒀을 때 (이거 되면 칠 준비 완료!!)
            if current_region[3]['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                print("----------------------------------------------")
                current_region[3]['counts'] += 1 #스타트 영역에 공이 있는지 없는지 실시간 판별
                current_region[3]['in'] = True    
                          
                if last_line_sound_count == 0:
                    threading.Thread(target=ready_putting).start()
                                                
                    last_line_sound_count += 1                            
                    first_line_second = current_time
                               
                if current_time - first_line_second > 6 and last_line_sound_count != 0:
                    
                    threading.Thread(target=ready_putting).start()

                    first_line_second = current_time
            if current_region[3]['in'] == True and  bbox_center[0] < list(current_region[3]['polygon'].exterior.coords)[0][0]:
                 print("########################################")
                 if golf_shot_sound == 0:
                    threading.Thread(target=golf_shot).start()
                    golf_shot_sound += 1


                          

        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            region_label = str(region['counts']) # 0,1,2,3...영역 안에 몇개가 들어있는지 나타냄.
            region_color = region['region_color']
            region_text_color = region['text_color']
            
            polygon_coords = np.array(region['polygon'].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region['polygon'].centroid.x), int(region['polygon'].centroid.y)

            text_size, _ = cv2.getTextSize(region_label, #region_label은 표현할 텍스트 여기서는 0,1,2,3 counts
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           fontScale=0.7,
                                           thickness=line_thickness)
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            #cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5),
            #             region_color, 1)
            #cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color,
            #            line_thickness)
            #print(region_label)
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow('Ultralytics YOLOv8 Region Counter Movable')
                cv2.setMouseCallback('Ultralytics YOLOv8 Region Counter Movable', mouse_callback)
            cv2.imshow('Ultralytics YOLOv8 Region Counter Movable', frame)
            

        if save_img:
            video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region['counts'] = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(japyo_lst)
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()
    pygame.quit()


def main():
    """Main function."""
    run()


if __name__ == '__main__':
    main()
