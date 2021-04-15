# # 该代码比较简单粗暴，参考了https://blog.csdn.net/shjsfx/article/details/105939059
#                           qishunwang.net/news_show_63515.aspx

# 修改了mp.solutions.drawing_utils内部函数

# 在第163-166行
#       if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
#         cv2.line(image, idx_to_coordinates[start_idx],
#                  idx_to_coordinates[end_idx], connection_drawing_spec.color,
#                  connection_drawing_spec.thickness)
# 后添加
#         cv2.line(img=image, pt1=idx_to_coordinates[4], pt2=idx_to_coordinates[8], color=(255, 0, 255), thickness=4)
#         pt = ((idx_to_coordinates[4][0]+idx_to_coordinates[8][0])//2,(idx_to_coordinates[4][1]+idx_to_coordinates[8][1])//2)
#         cv2.circle(img=image, center=pt, radius=8, color=(122,255,255), thickness=cv2.FILLED)
# 实现拇指点与食指点的连线，并计算其距离
#     dist = np.sqrt((idx_to_coordinates[4][0]-idx_to_coordinates[8][0])**2+(idx_to_coordinates[4][1]-idx_to_coordinates[8][1])**2)
# 最终让draw_landmarks函数返回距离
#   return dist
# 
# 
# 修改后的draw_landmarks函数：

def draw_landmarks(
    image: np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: List[Tuple[int, int]] = None,
    landmark_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: DrawingSpec = DrawingSpec()):
  """Draws the landmarks and the connections on the image.

  Args:
    image: A three channel RGB image represented as numpy ndarray.
    landmark_list: A normalized landmark list proto message to be annotated on
      the image.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected in the drawing.
    landmark_drawing_spec: A DrawingSpec object that specifies the landmarks'
      drawing settings such as color, line thickness, and circle radius.
    connection_drawing_spec: A DrawingSpec object that specifies the
      connections' drawing settings such as color and line thickness.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel RGB.
      b) If any connetions contain invalid landmark index.
  """
  if not landmark_list:
    return
  if image.shape[2] != RGB_CHANNELS:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape
  idx_to_coordinates = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < PRESENCE_THRESHOLD)):
      continue
    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   image_cols, image_rows)
    if landmark_px:
      idx_to_coordinates[idx] = landmark_px
  if connections:
    num_landmarks = len(landmark_list.landmark)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
        cv2.line(image, idx_to_coordinates[start_idx],
                 idx_to_coordinates[end_idx], connection_drawing_spec.color,
                 connection_drawing_spec.thickness)
        cv2.line(img=image, pt1=idx_to_coordinates[4], pt2=idx_to_coordinates[8], color=(255, 0, 255), thickness=4)
        pt = ((idx_to_coordinates[4][0]+idx_to_coordinates[8][0])//2,(idx_to_coordinates[4][1]+idx_to_coordinates[8][1])//2)
        cv2.circle(img=image, center=pt, radius=8, color=(122,255,255), thickness=cv2.FILLED)
    dist = np.sqrt((idx_to_coordinates[4][0]-idx_to_coordinates[8][0])**2+(idx_to_coordinates[4][1]-idx_to_coordinates[8][1])**2)

  # Draws landmark points after finishing the connection lines, which is
  # aesthetically better.
  for landmark_px in idx_to_coordinates.values():
    cv2.circle(image, landmark_px, landmark_drawing_spec.circle_radius,
               landmark_drawing_spec.color, landmark_drawing_spec.thickness)
  return dist

