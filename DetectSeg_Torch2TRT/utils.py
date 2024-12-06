from skimage import measure, draw, color, morphology
from skimage.measure import label, regionprops
import numpy as np
import cv2
import torch
import math

def mask_iou(mask1, mask2):
    mask1 = np.array(mask1).astype(np.int)
    mask2 = np.array(mask2).astype(np.int)
    mask1[mask1 > 0] = 1
    mask2[mask2 > 0] = 1
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)
    inter = np.sum(np.array((mask1+mask2)==2).astype(np.int))
    mask_iou = inter / (area1 + area2 - inter)
    return mask_iou


def head_MorAnalysis(head_map):
    pred_headmask = head_map.copy()
    label_img = label(pred_headmask.copy(), connectivity=head_map.ndim)
    label_img = morphology.remove_small_objects(label_img, min_size=50, connectivity=1)
    properties = measure.regionprops(label_img)
    if len(properties) > 0:
        prop = properties[0]
        perimeter = prop.perimeter
        a = prop.major_axis_length
        b = prop.minor_axis_length
        orientation = np.pi * 2 - float(prop.orientation)  # 方向为逆时针弧度，所以用2*pi-orien转换为顺时针
        yc, xc = prop.centroid
        # print('椭圆拟合结果：', (xc, yc), (a, b), orientation)

        ellipse = ((xc, yc), (b, a), orientation / np.pi * 180 % 360 + 90)
        box_elps = cv2.boxPoints(ellipse)  # 获取最小外接矩形的四个顶点坐标
        box_elps = np.int0(box_elps)
        pt1, pt2 = tuple(np.mean(box_elps[0:2], axis=0).astype(np.int)), tuple(
            np.mean(box_elps[2:4], axis=0).astype(np.int))
        pt3, pt4 = tuple(np.mean(box_elps[[0, 3]], axis=0).astype(np.int)), tuple(
            np.mean(box_elps[[1, 2]], axis=0).astype(np.int))

        # cal iou
        mask_ellipse = np.zeros_like(pred_headmask)
        cv2.ellipse(mask_ellipse, ellipse, 1, thickness=-1)
        iou = mask_iou(pred_headmask.copy(), mask_ellipse)
    else:
        ellipse = None
        pt3, pt4 = (0, 0), (0, 0)
        pt1, pt2 = (0, 0), (0, 0)
        perimeter = 0
        iou = 0
    return ellipse, (pt3, pt4), (pt1, pt2), perimeter, iou


# 对头部分割结果进行椭圆拟合
def ellipse_fit(crop_img):
    # 阈值处理
    _, thresh = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY)

    # 提取轮廓
    if cv2.__version__.split('.')[0] == '3':
        _, contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = []
    temp_len = []
    for idx, contour in enumerate(contours):
        if idx == 0:
            cnt = contour
            temp_len = len(contour)
        else:
            if temp_len < len(contour):
                cnt = contour
    # 椭圆拟合：cent_x, cent_y, a, b, 角度（以90度y轴正方向为0度，顺时针旋转计算度数）
    if len(cnt) > 5:
        ellipse = cv2.fitEllipse(cnt)
        len = cv2.arcLength(cnt, True)
    else:
        ellipse = None
        len = 0

    return ellipse, len


def vac_MorAnalysis(vac_pred, pt_major, pt_minor, head_coord, neck_coord, ellipse):
    # neck检测框[x1, y1, x2, y2] 中心点
    cent_x = (neck_coord[0] + neck_coord[2]) / 2
    cent_y = (neck_coord[1] + neck_coord[3]) / 2

    # 筛选出靠近颈部的major_point
    dist = []
    for idx, pt in enumerate(pt_major):
        pt = list(pt)
        pt[0] += head_coord[0]
        pt[1] += head_coord[1]
        dist.append((cent_x - pt[0]) ** 2 + (cent_y - pt[1]) ** 2)
    pt_major_idx = np.argmin(np.array(dist))

    # 求穿过点并平行于短轴的直线
    A, B, C = cal_ABC(pt_minor[0], pt_minor[1])
    A_ = -A / B
    B_ = -1
    C_ = pt_major[pt_major_idx][1] - A_ * pt_major[pt_major_idx][0]

    length = np.sqrt((pt_major[1][0] - pt_major[0][0]) ** 2 + (pt_major[1][1] - pt_major[0][1]) ** 2)

    # area_dmin_dmax_halfw_pos = []
    area_d_length_pos = []
    vac_label = label(vac_pred)
    vac_props = regionprops(vac_label)
    vac_num = 0
    vac_loc = 3
    if len(vac_props) == 0:
        area_d_length_pos.append([])
        vac_num = 0
        vac_loc = 3
    else:
        for vac_prop in vac_props:
            if vac_prop.area < 5:
                continue

            """方式1：所有点计算出最小值"""
            # ys, xs = np.nonzero(vac_label == vac_prop.label)
            # dmax = 0
            # dmin = 1e3
            # for x0, y0 in zip(xs, ys):
            #     d, _ = Point2Line((A_, B_, C_), (x0, y0))
            #     if d > dmax:
            #         dmax = d
            #     if d < dmin:
            #         dmin = d
            """方式2：计算质心位置"""

            y0, x0 = vac_prop.centroid
            d, _ = Point2Line((A_, B_, C_), (x0, y0))
            # d = dmin
            if d <= length / 3.:
                pos = 1
            elif d <= 2 * length / 3.:
                pos = 2
            else:
                pos = 3
            area_d_length_pos.append([vac_prop.area, d, length, pos])

            # update vac number and location
            vac_num += 1
            if 1 in [pos, vac_loc]:
                vac_loc = 1
            elif 2 in [pos, vac_loc]:
                vac_loc = 2
            else:
                vac_loc = 3

    return pt_major_idx, area_d_length_pos, vac_num, vac_loc


def cal_ABC(pt1, pt2):
    x1, y1= pt1
    x2, y2 = pt2
    if x2==x1:
        x2+=1e-8
    A = float(y2-y1) / float(x2-x1)
    B = -1
    C = float(x2*y1-x1*y2)/float(x2-x1)
    return A, B, C


def Point2Line(ABC, P):
    x0, y0 = P
    A, B, C = ABC
    d = np.abs(A*x0+B*y0+C).astype(np.float)/np.sqrt(A**2+B**2).astype(np.float)
    val = A*x0+B*y0+C
    return d, val


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter_area = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou

    return iou


def cal_dist(coord1, coord2, mini_dist):
    coord1 = np.array(coord1)[None]
    coord2 = np.array(coord2)[:, None]
    dist = coord1 - coord2
    dist = np.linalg.norm(dist, ord=2, axis=-1)
    min_val = np.min(dist)
    if min_val <= mini_dist:
        return True
    else:
        return False


def get_effective_residue_v0(msk_neck, msk_residue, mini_dist):

    msk_neck_f = msk_neck.copy()
    props_neck = regionprops(msk_neck)
    neck_coords = props_neck[0].coords

    msk_residue_label = label(msk_residue)
    props_residue = regionprops(msk_residue_label)
    for prop_residue in props_residue:
        lab = prop_residue.label
        tmp_coords = prop_residue.coords
        if cal_dist(neck_coords, tmp_coords, mini_dist):
            msk_neck_f[msk_residue_label == lab] = 2
    return msk_neck_f



def get_cnt(msk):
    """获取掩码的轮廓"""
    if cv2.__version__.split('.')[0] == '3':
        _, contours, hierarchy = cv2.findContours(msk.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(msk.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_effective_residue(msk_neck, msk_residue, mini_dist):
    """返回距离颈部距离小于mini_dist的胞浆区域"""
    residue_tmp = np.zeros_like(msk_residue).astype(np.uint8)

    """获取颈部的轮廓"""
    cnts_neck = get_cnt(msk_neck)
    if len(cnts_neck) == 0:
        return residue_tmp[None, :, :]
    else:
        neck_coords = cnts_neck[0]
        shape_ = cnts_neck[0].shape
        neck_coords = neck_coords.reshape(shape_[0], -1)

    """计算不同区域胞浆轮廓 到 颈部 的距离，若小于mini_dist则保存"""
    cnts_residue = get_cnt(msk_residue)
    for cnt_idx, cnt in enumerate(cnts_residue):
        residue_coords = cnt.reshape(cnt.shape[0], -1)
        if cal_dist(neck_coords, residue_coords, mini_dist):
            cv2.drawContours(residue_tmp, cnts_residue, cnt_idx, 1, -1)

    return residue_tmp[None, :, :]


"""
neck residue parameters calculation
"""
# 提取轮廓
def get_contours(gray_img):
    # 提取轮廓
    if cv2.__version__.split('.')[0] == '3':
        _, contours, hierarchy = cv2.findContours(gray_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(gray_img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# 计算颈部长度
def get_neck_len(p1, p2, p3, p4):
    h1_x = float((p1[0] + p2[0]) / 2)
    h1_y = float((p1[1] + p2[1]) / 2)
    h2_x = float((p4[0] + p3[0]) / 2)
    h2_y = float((p4[1] + p3[1]) / 2)
    tmp1 = float(math.sqrt((h1_x - h2_x) * (h1_x - h2_x) + (h1_y - h2_y) * (h1_y - h2_y)))  # 颈部长度
    tmp_pts1 = ((h1_x, h1_y), (h2_x, h2_y))

    h1_x = float((p3[0] + p2[0]) / 2)
    h1_y = float((p3[1] + p2[1]) / 2)
    h2_x = float((p4[0] + p1[0]) / 2)
    h2_y = float((p4[1] + p1[1]) / 2)
    tmp2 = float(math.sqrt((h1_x - h2_x) * (h1_x - h2_x) + (h1_y - h2_y) * (h1_y - h2_y)))  # 颈部长度
    if tmp1 > tmp2:
        return tmp1, tmp_pts1[0], tmp_pts1[1]
    else:
        return tmp2, (h1_x, h1_y), (h2_x, h2_y)


# 获取长度的垂线与轮廓的交点
def get_neck_wide(c1, c2, contour):
    k_c = (c1[1] - c2[1]) / (c1[0] - c2[0] + 1e-8)  # 长度中线斜率  中线公式y = k_h*(x-h1_x)+h1_y
    k_w = -1 / (k_c + 1e-8)  # 中线的垂线的斜率

    h_x = c1[0] + 0.5  # 取初始垂线上的点
    points = []
    while True:
        if h_x < c2[0]:
            # 计算中线与垂线的交点
            y_h = k_c * (h_x - c1[0]) + c1[1]
            # 交点坐标 point = (h_x, y_h)
            # 中线的垂线公式 y = k_w*(x-h1_x_1)+y_h
            cross = set()
            for con in contour:
                x, y = con[0][0], con[0][1]
                if int(y) == int(k_w * (x - h_x) + y_h):
                    # print(f'x:{x},y:{y}')
                    cross.add((x, y))
            cross = list(cross)
            if len(cross) == 2:  # 取交点为2个的垂线
                points.append(cross)
            h_x += 0.5
        else:
            break
    # print('points list:', points)
    return points


# 获取颈部数据
def get_neck_len_wid(box):
    p1, p2, p3, p4 = box[0], box[1], box[2], box[3]
    # 查找矩形的长所在直线的两个点
    point1 = p1
    l1 = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
    l2 = math.sqrt((p1[0] - p4[0]) * (p1[0] - p4[0]) + (p1[1] - p4[1]) * (p1[1] - p4[1]))
    if l1 > l2:
        point2 = p2
        return l1, l2, (point1, point2)
    else:
        point2 = p4
        return l2, l1, (point1, point2)


def get_neck_data(img_data, is_show=False):
    contours = get_contours(img_data)
    max_num = 0
    if len(contours) == 0:
        return 0, 0, (0, 0)

    for cnt_tmp in contours:
        if cnt_tmp.shape[0] > max_num:
            cnt = cnt_tmp
            max_num = cnt.shape[0]

    # 最小外接矩形:中心点坐标，最小外接矩形宽高和倾斜角度 x, y, w, h, angle = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    rect = cv2.minAreaRect(cnt)

    # 获取最小外接矩形的四个顶点坐标
    points_rect = cv2.boxPoints(rect)
    box = np.int0(points_rect)  # 将坐标转成整数 [[21 53] [31 47] [51 80] [41 86]]

    # 查找长度的中线:即颈部长度
    h_len, h_wid, neck_points = get_neck_len_wid(box)

    return h_len, h_wid, neck_points


# 获取颈部的数据
def get_neck_data_raw(img_data, is_show=False):
    contours = get_contours(img_data)
    max_num = 0
    if len(contours) == 0:
        return 0, 0

    for cnt_tmp in contours:
        if cnt_tmp.shape[0] > max_num:
            cnt = cnt_tmp
            max_num = cnt.shape[0]

    # 最小外接矩形:中心点坐标，最小外接矩形宽高和倾斜角度 x, y, w, h, angle = rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]
    rect = cv2.minAreaRect(cnt)

    # 获取最小外接矩形的四个顶点坐标
    points_rect = cv2.boxPoints(rect)
    box = np.int0(points_rect)  # 将坐标转成整数 [[21 53] [31 47] [51 80] [41 86]]
    # img_minarea = cv2.drawContours(img, [box], -1, (0, 255, 0), 1)
    p1, p2, p3, p4 = box[0], box[1], box[2], box[3]
    # print(p1, p2, p3, p4)

    # 查找长度的中线:即颈部长度
    h_len, c1, c2 = get_neck_len(p1, p2, p3, p4)

    # 计算颈部平均宽度
    points = get_neck_wide(c1, c2, cnt)
    if len(points) > 1:
        total_distance = 0
        for tow_point in points:
            point1, point2 = tow_point[0], tow_point[1]
            # print(f'po1:{point1},po2:{point2}')  # po1:(24, 57),po2:(33, 52)
            distance = int(math.sqrt((point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (
                    point1[1] - point2[1])))
            # print('distance:',distance)
            total_distance += distance
        mean_wide = int(total_distance / len(points))
    else:
        mean_wide = 0

    if h_len > mean_wide:
        if is_show:
            print(f'颈部长度为：{h_len * 2}')
            print(f'颈部平均宽度为：{mean_wide * 2}')
            print('-' * 30)
        return h_len * 2, mean_wide * 2
    else:
        if is_show:
            print(f'颈部长度为：{mean_wide * 2}')
            print(f'颈部平均宽度为：{h_len * 2}')
            print('-' * 30)
        return mean_wide * 2, h_len * 2


# 获取包浆的数据
def get_residue_data(img_data, is_show=False):
    contours = get_contours(img_data)
    if len(contours) == 0:
        return 0

    total_area = 0
    for contour in contours:
        # 获取包浆面积
        residue_area = cv2.contourArea(contour)
        total_area += residue_area

    if is_show:
        print(f'胞浆面积为:{total_area}')
        print('-' * 30)
    return total_area * 4


# 根据已知两点坐标，求过这两点的直线解析方程： a*x+b*y+c = 0  (a >= 0)
def get_line(points):
    p1x, p1y, p2x, p2y = points[0][0], points[0][1], points[1][0], points[1][1]
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)

    return [a, b, c]


# 获取头部线段与颈部平行线段的夹角
def cal_angle_v0(head_points, neck_points, head_coord, neck_coord):
    """根据头部和颈部检测框，获得头部、颈部点的坐标"""
    head_points = list(head_points)
    neck_points = list(neck_points)
    for idx, pt in enumerate(head_points):
        pt = list(pt)
        pt[0] += head_coord[0]
        pt[1] += head_coord[1]
        head_points[idx] = pt
    print(f'Head_points:{head_points}')

    for idx, pt in enumerate(neck_points):
        pt = list(pt)
        pt[0] += neck_coord[0]
        pt[1] += neck_coord[1]
        neck_points[idx] = pt
    print(f'Neck_points:{neck_points}')

    n_k = (neck_points[1][1] - neck_points[0][1]) / (neck_points[1][0] - neck_points[0][0] + 1e-8)
    n_b = neck_points[0][1] - n_k * neck_points[0][0]

    # 颈部和头部的4个点
    n_1x, n_1y = neck_points[0][0], neck_points[0][1]
    n_2x, n_2y = neck_points[1][0], neck_points[1][1]
    h_1x, h_1y = head_points[0][0], head_points[0][1]
    h_2x, h_2y = head_points[1][0], head_points[1][1]
    # 4个点之间每2个点之间的距离
    d_n1_h1 = int(math.sqrt((n_1x - h_1x) * (n_1x - h_1x) + (n_1y - h_1y) * (n_1y - h_1y)))
    d_n2_h1 = int(math.sqrt((n_2x - h_1x) * (n_2x - h_1x) + (n_2y - h_1y) * (n_2y - h_1y)))
    d_n1_h2 = int(math.sqrt((n_1x - h_2x) * (n_1x - h_2x) + (n_1y - h_2y) * (n_1y - h_2y)))
    d_n2_h2 = int(math.sqrt((n_2x - h_2x) * (n_2x - h_2x) + (n_2y - h_2y) * (n_2y - h_2y)))

    # 根据颈部各点到头部各点的距离判断，头部哪个点靠近颈部
    min_d = min(d_n1_h1, d_n2_h1, d_n1_h2, d_n2_h2)
    n_x1, n_y1 = 0, 0
    if min_d == d_n1_h1 or min_d == d_n2_h1:
        n_x1, n_y1 = h_1x, h_1y
    elif min_d == d_n1_h2 or min_d == d_n2_h2:
        n_x1, n_y1 = h_2x, h_2y

    # 颈部平行直线
    n_k2 = n_k
    n_b2 = n_y1 - n_k2 * n_x1
    # 头部的长
    d = int(math.sqrt((head_points[0][0] - head_points[1][0]) * (head_points[0][0] - head_points[1][0]) + (
            head_points[0][1] - head_points[1][1]) * (head_points[0][1] - head_points[1][1])))
    # 颈部平行线上，与头部长相等的线段的另一端
    n_x2 = int(math.sqrt(d * d / (1 + n_k2 * n_k2)) + n_x1)
    n_y2 = int(n_k2 * n_x2 + n_b2)

    # 判断线段另一端到颈部中点的距离是否大于线段本身的长
    # 颈部2点的中点
    center_n = [int((neck_points[0][0] + neck_points[1][0]) / 2), int((neck_points[0][1] + neck_points[1][1]) / 2)]
    center_n = tuple(center_n)
    # print(center_n)

    # 线段另一端到中点的距离
    d2 = int(math.sqrt((center_n[0] - n_x2) * (center_n[0] - n_x2) + (center_n[1] - n_y2) * (center_n[1] - n_y2)))
    # 判断线段与d2的长度
    if d2 > d:
        n_x2 = int(-math.sqrt(d * d / (1 + n_k2 * n_k2)) + n_x1)
        n_y2 = int(n_k2 * n_x2 + n_b2)

    # 计算夹角
    angle1 = math.atan2(head_points[0][1] - head_points[1][1], head_points[0][0] - head_points[1][0])
    angle2 = math.atan2(n_y2 - n_y1, n_x2 - n_x1)
    angleDegrees = int((angle1 - angle2) * 360 / (2 * math.pi))
    # 夹角为负数时，变为正数
    if angleDegrees < 0:
        angleDegrees = -angleDegrees
    print(f'angleDegrees:{angleDegrees}')

    return angleDegrees


def cal_angle(head_pts, neck_pts, head_coord, neck_coord, pt_major_idx):
    """根据头部和颈部检测框，获得头部、颈部点的坐标"""
    head_pts = list(head_pts)
    neck_pts = list(neck_pts)
    for idx, pt in enumerate(head_pts):
        pt = list(pt)
        pt[0] += head_coord[0]
        pt[1] += head_coord[1]
        head_pts[idx] = pt

    for idx, pt in enumerate(neck_pts):
        pt = list(pt)
        pt[0] += neck_coord[0]
        pt[1] += neck_coord[1]
        neck_pts[idx] = pt

    # 将靠近颈部的点放到位置0
    if pt_major_idx == 1:
        head_pts = (head_pts[1], head_pts[0])
    # print(f'Head_points:{head_pts}')

    # 计算颈部哪个点离上面的更近
    dist_square = []
    for idx, neck_pt in enumerate(neck_pts):
        dist_square.append(sum([(i - j) * (i - j) for i, j in zip(neck_pt, head_pts[0])]))
    if dist_square[1] < dist_square[0]:
        neck_pts = (neck_pts[1], neck_pts[0])
    # print(f'Neck_points:{neck_pts}')

    # 计算角度
    head_vec = [i - j for i, j in zip(head_pts[1], head_pts[0])]
    neck_vec = [i - j for i, j in zip(neck_pts[1], neck_pts[0])]
    head_norm = math.sqrt(sum([i * i for i in head_vec]))
    neck_norm = math.sqrt(sum([i * i for i in neck_vec]))
    angle = math.acos(sum([i * j for i, j in zip(head_vec, neck_vec)]) / head_norm / neck_norm) * 180 / np.pi
    return angle


class SingleImageConclusion:
    """通过单张图像各部分参数，对精子头部、空泡、颈部、胞浆进行判定,输出该张精子图像的判定结果"""
    def __init__(self, item):
        """
        :param head_len: 头部长度
        :param head_wid: 头部宽度
        :param head_len_wid_ratio:头部长宽比
        :param vacs_head_area_ratio: 空泡占头部面积的比例
        :param vacs_num: 空泡数量
        :param vacs_loc: 空泡位置
        :param neck_wid: 颈部宽度
        :param neck_head_angle:颈部与头部的夹角
        :param rsd_head_area_ratio: 胞浆占头部面积的比例
        """
        head_len, head_wid, head_len_wid_ratio, vacs_head_area_ratio, vacs_num, \
        vacs_loc, neck_wid, neck_head_angle, rsd_head_area_ratio = item
        print(item)
        self.factor = 1
        self.head_len = head_len / self.factor
        self.head_wid = head_wid / self.factor
        self.head_len_wid_ratio = head_len_wid_ratio
        self.vacs_head_area_ratio = vacs_head_area_ratio
        self.vacs_num = int(vacs_num)
        self.vacs_loc = int(vacs_loc)
        self.neck_wid = neck_wid / self.factor
        self.neck_head_angle = float(neck_head_angle)
        self.rsd_head_area_ratio = rsd_head_area_ratio
        self.single_conclusion = []

    def head_conclusion(self):
        """
        头部判定

        头部长度
        正常n：4~6
        偏正常p：3.4~4；6~6.9
        异常a：＞=6.9；＜=3.4

        头部宽度
        正常n：2.5~4
        偏正常p：2.1~2.5；4~4.6
        异常a：＞=4.6；＜=2.1

        头部长宽比
        正常n：1.2~1.8
        异常a：＞=1.8；＜=1.2
        :return:
        """
        head_list = []
        # 特例
        if int(self.head_len) == 0 and int(self.head_wid) == 0 \
            and int(self.head_len_wid_ratio) == 0:
            return 'u'

        # 头部长度的判定
        if 4 < self.head_len < 6:
            head_list.append('n')
        elif 3.4 < self.head_len <= 4 or 6 <= self.head_len < 6.9:
            head_list.append('p')
        elif self.head_len <= 3.4 or self.head_len >= 6.9:
            head_list.append('a')

        # 头部宽度的判定
        if 2.5 < self.head_wid < 4:
            head_list.append('n')
        elif 2.1 < self.head_wid <= 2.5 or 4 <= self.head_wid < 4.6:
            head_list.append('p')
        elif self.head_wid <= 2.1 or self.head_wid >= 4.6:
            head_list.append('a')

        # 头部长宽比的判定
        if 1.2 < self.head_len_wid_ratio < 1.8:
            head_list.append('n')
        elif self.head_len_wid_ratio <= 1.2 or self.head_len_wid_ratio >= 1.8:
            head_list.append('a')

        if 'u' in head_list:
            return 'u'
        elif 'a' in head_list:
            return 'a'
        elif 'p' in head_list:
            return 'p'
        else:
            return 'n'

    def vacs_conclusion(self):
        """
        空泡判定

        空泡占头部面积比例
        正常：0；
        偏正常：＜5%；
        异常：＞=5%

        空泡数量
        正常：0
        偏正常：≤2；
        异常：>=3

        空泡位置
        正常：上1/3 (3)
        偏正常：中1/3 (2)
        异常：下1/3 (1)
        :return:
        """
        vacs_list = []
        # 空泡占头部面积比例的判定
        if self.vacs_head_area_ratio == 0:
            vacs_list.append('n')
        elif self.vacs_head_area_ratio < 0.05:
            vacs_list.append('p')
        elif self.vacs_head_area_ratio >= 0.05:
            vacs_list.append('a')

        # 空泡数量的判定
        if self.vacs_num == 0:
            vacs_list.append('n')
        elif self.vacs_num <= 2:
            vacs_list.append('p')
        elif self.vacs_num > 2:
            vacs_list.append('a')

        # 空泡位置的判定
        if self.vacs_loc == 3:
            vacs_list.append('n')
        elif self.vacs_loc == 2:
            vacs_list.append('p')
        elif self.vacs_loc == 1:
            vacs_list.append('a')

        if 'u' in vacs_list:
            return 'u'
        elif 'a' in vacs_list:
            return 'a'
        elif 'p' in vacs_list:
            return 'p'
        else:
            return 'n'

    def neck_conclusion(self):
        """
        颈部判定

        颈部宽度
        正常：<2.5
        异常：>=2.5

        头部与颈部夹角
        正常：150~180
        异常：=<150
        :return:
        """
        # 特例
        if self.neck_head_angle < 0 and int(self.neck_wid) == 0:
            return 'u'

        neck_list = []
        # 颈部宽度的判定
        if int(self.neck_wid) == 0:
            neck_list.append('u')
        elif self.neck_wid < 2.5:
            neck_list.append('n')
        elif self.neck_wid >= 2.5:
            neck_list.append('a')

        # 头部与颈部夹角的判定
        if self.neck_head_angle < 0:
            neck_list.append('u')
        elif 150 < self.neck_head_angle <= 180:
            neck_list.append('n')
        elif self.neck_head_angle <= 150:
            neck_list.append('a')

        if 'u' in neck_list:
            return 'u'
        elif 'a' in neck_list:
            return 'a'
        elif 'p' in neck_list:
            return 'p'
        else:
            return 'n'

    def residue_conclusion(self):
        """
        胞浆判定

        包浆占头部比例
        正常：0
        偏正常：<1/3
        异常：>=1/3
        :return:
        """
        residue_list = []
        # 包浆占头部比例的判定
        if self.rsd_head_area_ratio == 0:
            residue_list.append('n')
        elif self.rsd_head_area_ratio < 1 / 3.0:
            residue_list.append('p')
        elif self.rsd_head_area_ratio >= 1 / 3.0:
            residue_list.append('a')

        if 'a' in residue_list:
            return 'a'
        elif 'p' in residue_list:
            return 'p'
        else:
            return 'n'

    def image_conclusion(self):
        """

        :return: ['头部判定', '空泡判定', '颈部判定', '胞浆判定', '该张图像精子的判定']
        """
        head_con = self.head_conclusion()
        vacs_con = self.vacs_conclusion()
        neck_con = self.neck_conclusion()
        residue_con = self.residue_conclusion()
        self.single_conclusion.append(head_con)
        self.single_conclusion.append(vacs_con)
        self.single_conclusion.append(neck_con)
        self.single_conclusion.append(residue_con)

        if 'u' in self.single_conclusion:
            self.single_conclusion.append('u')
        elif 'a' in self.single_conclusion:
            self.single_conclusion.append('a')
        elif 'p' in self.single_conclusion:
            self.single_conclusion.append('p')
        else:
            self.single_conclusion.append('n')

        return self.single_conclusion
