import cv2
import numpy as np
import time
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Function to draw a traffic light
def draw_traffic_light(img, state):
    rect_x, rect_y = 725, 42 # Position of the rectangle
    rect_width, rect_height = 150, 420 # Size of the rectangle
    border_radius = 20 # Radius of rounded borders
    rectangle_color = (169, 169, 169) # Color of the rectangle
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), rectangle_color, -1)
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (169, 169, 169), 3, cv2.LINE_AA)
    cv2.rectangle(img, (rect_x + border_radius, rect_y), (rect_x + rect_width - border_radius, rect_y + rect_height), (169, 169, 169), -1)
    cv2.rectangle(img, (rect_x, rect_y + border_radius), (rect_x + rect_width, rect_y + rect_height - border_radius), (169, 169, 169), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,f"State: {state}", (713, 500),font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # Change state
    if state == 'red':
        cv2.circle(img, (800, 100), 50, (0, 0, 255), -1)
        cv2.circle(img, (800, 250), 50, (0, 0, 0), -1)
        cv2.circle(img, (800, 400), 50, (0, 0, 0), -1)
    elif state == 'yellow':
        cv2.circle(img, (800, 250), 50, (0, 255, 255), -1)
        cv2.circle(img, (800, 100), 50, (0, 0, 0), -1)
        cv2.circle(img, (800, 400), 50, (0, 0, 0), -1)
    elif state == 'green':
        cv2.circle(img, (800, 400), 50, (0, 255, 0), -1)
        cv2.circle(img, (800, 250), 50, (0, 0, 0), -1)
        cv2.circle(img, (800, 100), 50, (0, 0, 0), -1)

# Function to draw digital timer
def draw_timer(img, seconds):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"Time: {seconds}", (20, 480), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
def light(light_state,distance,distance_2,moved_from_eye1,moved_from_eye2):
    if distance <= 3 and not moved_from_eye1 and moved_from_eye2: # Moving in radius of 1 vision eye.
        light_state = 'yellow'
    elif distance_2 <= 3 and moved_from_eye1 and not moved_from_eye2: # Moving in radius of 2 vision eye.
        light_state = 'yellow'
    elif distance_2 <= 3 and not moved_from_eye1 and moved_from_eye2: # Moving out of radius of 1 vision eye.
        light_state = 'yellow'
    elif distance_2 <= 3 and not moved_from_eye1 and moved_from_eye2: # Moving out of radius of 2 vision eye.
        light_state = 'yellow'
    elif not moved_from_eye1 and moved_from_eye2: # Moving towards 1 vision eye.
        light_state = 'red'
    elif moved_from_eye1 and not moved_from_eye2: # Moving towards 2 vision eye.
        light_state = 'red'
    else:
        light_state = 'green'
    return light_state
        
def direction(moved_from_eye1,moved_from_eye2): 
    if(moved_from_eye1 and moved_from_eye2):
        print(f"Object is moving away from crosswalk")
    elif(moved_from_eye1 and not moved_from_eye2):
        print(f"Object is moving towards right end of crosswalk")
    elif(not moved_from_eye1 and  moved_from_eye2):
        print(f"Object is moving towards left end of crosswalk")
    else:
        print(f"Object is moving towards crosswalk")

# Main function
def main():
    cap = cv2.VideoCapture("vid1.mp4")
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('visioneye-distance-calculation.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    pixel_per_meter = 70
    txt_color, txt_background, bbox_clr = ((0, 0, 0), (255, 255, 255), (255, 0, 255))
    txt_color_2, txt_background_2, bbox_clr_2 = ((119, 7, 55), (255, 182, 193), (255, 0, 255))
    # Video capture
    cap = cv2.VideoCapture('vid1.mp4') # Replace 'your_video.mp4' with your video file

    # Dictionary to store tracking IDs and coordinates
    prev_coordinates = {}
    distance = 0
    distance_2 = 0    
    
    # Traffic light state
    light_state = 'green'
    
    # Timer
    start_time = time.time()
    elapsed_time = 0
    height = 500
    width = 1000
    moved_from_eye1 = False
    moved_from_eye2 = False

    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        
        # Check if video opened successfully
        if not cap.isOpened():
            print("Error: Video file not found or cannot be opened.")
            return
        elapsed_time = int(time.time() - start_time)
        annotator = Annotator(im0, line_width=2)
        model1 = YOLO('runs/detect/train3/weights/best.pt', task='detect', verbose=False)
        model2 = YOLO("yolov8n.pt")
        
        results2 = model2.track(im0, persist=True, classes=[0])
        results1 = model1(im0,show=False,conf=0.4,save=False)
    
        boxes1 = results1[0].boxes.xyxy.cpu()
        
        boxes2 = results2[0].boxes.xyxy.cpu()
        # print(boxes1)
        
        center_point = ((int(boxes1[0][0].item()+boxes1[0][2].item()/2))-20, int(boxes1[0][1].item()))
        center_point_2 = ((int(boxes1[0][0].item()+boxes1[0][2].item()/2)), int(boxes1[0][3].item()))  # New center point (adjust coordinates as needed
        # print(center_point)
        # print(center_point_2)
        boxes = np.concatenate((boxes1, boxes2), axis=0)

        if results1[0].boxes.id is not None:
            track_ids = results1[0].boxes.id.int().cpu().tolist()
        if results2[0].boxes.id is not None:
            track_ids = results2[0].boxes.id.int().cpu().tolist()
            valid = True
            for box, track_id in zip(boxes, track_ids):
                annotator.box_label(box, label=str(track_id), color=bbox_clr)
                print(str(track_id))
                if(box in boxes1.cpu().numpy()):
                    continue
                
                annotator.visioneye(box, center_point)
                annotator.visioneye(box, center_point_2)

                x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)    # Bounding box centroid
                
                
                if track_id in prev_coordinates:
                    distance = math.sqrt(math.fabs(x1 - center_point[0])**2 + math.fabs(y1 - center_point[1])**2) / pixel_per_meter
                    distance_2 = math.sqrt(math.fabs(x1 - center_point_2[0])**2 + math.fabs(y1 - center_point_2[1])**2) / pixel_per_meter
                    
                    # Check if the object moved away from one or both vision eyes
                    moved_from_eye1 = distance > prev_coordinates[track_id][0]
                    moved_from_eye2 = distance_2 > prev_coordinates[track_id][1]
                    
                    # direction(moved_from_eye1,moved_from_eye2)
                    if distance <= 2 or  distance_2 <= 2: # Moving in radius of 1 vision eye.
                        valid = False
                        # light_state = 'yellow'
                        # light_state='red'
                    # elif not moved_from_eye1 and moved_from_eye2: # Moving towards 1 vision eye.
                    #     light_state = 'red'
                    # elif moved_from_eye1 and not moved_from_eye2: # Moving towards 2 vision eye.q
                    #     light_state = 'red'
                    # else:
                    #     valid = True
                        # light_state = 'green'
                # Update previous coordinates
                if not valid:
                    start_time=time.time()
                    elapsed_time=0
                    if elapsed_time>=0 and elapsed_time<=5:
                        light_state="yellow"
                        light_state= "red"
                    if elapsed_time>14:
                        light_state="green"
                else:
                    light_state= "green"
                prev_coordinates[track_id] = (distance, distance_2)

                text_size, _ = cv2.getTextSize(f"Distance: {distance:.2f} m", cv2.FONT_HERSHEY_SIMPLEX,1.2, 3)
                cv2.rectangle(im0, (x1, y1 - text_size[1] - 10),(x1 + text_size[0] + 10, y1), txt_background, -1)
                cv2.putText(im0, f"Distance: {distance:.2f} m",(x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2,txt_color, 3)
                
                text_size, _ = cv2.getTextSize(f"Distance: {distance_2:.2f} m", cv2.FONT_HERSHEY_SIMPLEX,1.2, 3)
                cv2.rectangle(im0, (x1+60, y1+60 - text_size[1] - 10),(x1+60 + text_size[0] + 10, y1+60), txt_background_2, -1)
                cv2.putText(im0, f"Distance: {distance_2:.2f} m",(x1+60, y1+60 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2,txt_color_2, 3)
                
                # Create empty canvas
                canvas = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
                border_color = (0, 0, 0)
                
                # light(light_state,distance,distance_2,moved_from_eye1,moved_from_eye2)
                
        # Create empty canvas
        canvas = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)
        border_color = (0, 0, 0)

        # Draw border
        border_thickness = 1
        cv2.rectangle(canvas, (0, 0), (width, height-1), border_color, border_thickness)

        # im0 = imutils.resize(im0, width=800)  # Resize the image
        im0 = cv2.resize(im0, (600, 500))
        # height, width = im0.shape[:2]  # Get image height and width
        canvas = np.full((height, width, 3), (255, 255, 255), dtype=np.uint8)  # Create canvas with image dimensions
        border_color = (0, 0, 0)
        
        # Draw border
        border_thickness = 1
        cv2.rectangle(canvas, (0, 0), (width+100, height-1), border_color, border_thickness)
        # Place video frame on the left
        canvas[:500, :600] = im0
        draw_traffic_light(canvas, light_state)

        # Draw digital timer at the bottom

        draw_timer(canvas, elapsed_time)

        # Show canvas
        cv2.imshow("Okay",canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()