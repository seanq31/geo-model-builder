import random

def read_GMBL_list(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    return lines


def get_name_special_param(obj_type, reader):
    if reader is None or reader.points is None:
        cnt = 0
    else:
        cnt = len(reader.points)

    if obj_type in ['acute-tri', 'acute-iso-tri', 'iso-tri', 'right-tri', 'triangle']:
        obj_name = "".join(['(P', str(cnt + 1),
                            ' P', str(cnt + 2), 
                            ' P', str(cnt + 3),
                            ')'])
    elif obj_type == 'polygon':
        rand_num = random.sample(range(6), 1)[0] + 3
        obj_name = '('
        for i in range(rand_num):
            obj_name += 'P' + str(cnt + i + 1) + ' '
        obj_name = obj_name[: -1]
        obj_name += ')'

    if obj_type in ['acute-iso-tri', 'iso-tri', 'right-tri']:
        samp = 'P' + str(random.sample(range(3), 1)[0] + 1 + cnt)
        obj_type = ''.join(['(', obj_type, ' ', samp, ')'])

    return [obj_name, obj_type]


def specify_relation(relation, reader):
    if reader is None or reader.points is None:
        cnt_points_reader = 0
    else:
        cnt_points_reader = len(reader.points)
    if reader is None or reader.all_lines is None:
        cnt_lines_reader = 0
    else:
        cnt_lines_reader = len(reader.all_lines)
    if reader is None or reader.circles is None:
        cnt_circles_reader = 0
    else:
        cnt_circles_reader = len(reader.circles)
    
    list_objs = None

    # predicates, for assert or eval 
    if relation == 'centroid':
        list_objs = ['point'] * 4
    elif relation == 'concur':
        list_objs = ['line'] * 3
    elif relation == 'circumcenter':
        list_objs = ['point'] * 4
    elif relation == 'cong':
        list_objs = ['point'] * 4
    elif relation == 'contri':
        list_objs = ['point'] * 6
    elif relation == 'coll':
        list_objs = ['point'] * 3
    elif relation == 'cycl':
        if cnt_points_reader >= 4:
            num = random.sample(range(4, cnt_points_reader + 1), 1)[0]
        else:
            num = 4
        list_objs = ['point'] * num
    elif relation == '=':
        rand_num = random.uniform(0, 1)
        if cnt_points_reader >= 2 and rand_num < 0.7:
            list_objs = ['point'] * 2
        elif cnt_lines_reader >= 2:
            list_objs = ['line'] * 2
    elif relation == 'eq-ratio':
        list_objs = ['point'] * 8
    elif relation == 'foot':
        list_objs = ['point'] * 2 + ['line']   
    elif relation == 'incenter':
        list_objs = ['point'] * 4
    elif relation == 'inter-ll':
        list_objs = ['point'] + ['line'] * 2
    elif relation == 'midp':
        list_objs = ['point'] * 3
    elif relation == 'on-circ':
        list_objs = ['point'] + ['circle']
    elif relation == 'on-line':
        list_objs = ['point'] + ['line']
    elif relation == 'on-ray':
        list_objs = ['point'] * 3
    elif relation == 'on-seg':
        list_objs = ['point'] * 3
    elif relation == 'opp-sides':
        list_objs = ['point'] * 2 + ['line']
    elif relation == 'orthocenter':
        list_objs = ['point'] * 4
    elif relation == 'perp':
        list_objs = ['line'] * 2
    elif relation == 'para':
        list_objs = ['line'] * 2
    elif relation == 'same-side':
        list_objs = ['point'] * 2 + ['line']
    elif relation == 'sim-tri':
        list_objs = ['point'] * 6
    elif relation == 'tangent-cc':
        list_objs = ['circle'] * 2
    elif relation == 'tangent-lc':
        list_objs = ['line'] + ['circle']
    elif relation == 'tangent-at-cc':
        list_objs = ['point'] + ['circle'] * 2
    elif relation == 'tangent-at-lc':
        list_objs = ['point'] + ['line'] + ['circle']

    # functions, for define
    if relation == 'amidp-opp':
        list_objs = ['point'] * 3
    elif relation == 'amidp-same':
        list_objs = ['point'] * 3
    elif relation == 'centroid':
        list_objs = ['point'] * 3
    elif relation == 'circumcenter':
        list_objs = ['point'] * 3
    elif relation == 'excenter':
        list_objs = ['point'] * 3
    elif relation == 'foot':
        list_objs = ['point'] + ['line']
    elif relation == 'harmonic-conj':
        list_objs = ['point'] * 3
    elif relation == 'incenter':
        list_objs = ['point'] * 3
    elif relation == 'inter-cc':
        list_objs = ['circle'] * 2 + ['root-selector']
    elif relation == 'inter-ll':
        list_objs = ['line'] * 2
    elif relation == 'inter-lc':
        list_objs = ['line'] + ['circle'] + ['root-selector']
    elif relation == 'isogonal-conj':
        list_objs = ['point'] * 4
    elif relation == 'isotomic-conj':
        list_objs = ['point'] * 4
    elif relation == 'midp':
        list_objs = ['point'] * 2
    elif relation == 'mixtilinear-incenter':
        list_objs = ['point'] * 3
    elif relation == 'orthocenter':
        list_objs = ['point'] * 3
    elif relation == 'line':
        list_objs = ['point'] * 2
    elif relation == 'isogonal':
        list_objs = ['point'] * 4
    elif relation == 'isotomic':
        list_objs = ['point'] * 4
    elif relation == 'perp-bis':
        list_objs = ['point'] * 2
    elif relation == 'perp-at':
        list_objs = ['point'] + ['line']
    elif relation == 'c3':
        list_objs = ['point'] * 3
    elif relation == 'circumcircle':
        list_objs = ['point'] * 3
    elif relation == 'excircle':
        list_objs = ['point'] * 3
    elif relation == 'incircle':
        list_objs = ['point'] * 3
    elif relation == 'mixtilinear-incircle':
        list_objs = ['point'] * 3
    elif relation == 'diam':
        list_objs = ['point'] * 2
    
    # parameterizations, for param
    if relation == 'on-circ':
        list_objs = ['circle']
    elif relation == 'on-line':
        list_objs = ['line']
    elif relation == 'on-major-arc':
        list_objs = ['circle'] + ['point'] * 2
    elif relation == 'on-minor-arc':
        list_objs = ['circle'] + ['point'] * 2
    elif relation == 'in-poly':
        if cnt_points_reader >= 3:
            num = random.sample(range(3, cnt_points_reader + 1), 1)[0]
        else:
            num = 3
        list_objs = ['point'] * num
    elif relation == 'on-ray':
        list_objs = ['point'] * 2
    elif relation == 'on-ray-opp':
        list_objs = ['point'] * 2
    elif relation == 'on-seg':
        list_objs = ['point'] * 2
    elif relation == 'tangent-lc':
        list_objs = ['circle']
    elif relation == 'through':
        list_objs = ['point']
    elif relation == 'tangent-cc':
        list_objs = ['circle']
    elif relation == 'tangent-cl':
        list_objs = ['line']
    elif relation == 'origin':
        list_objs = ['point']

    # root selector
    if relation == 'rs-neq':
        list_objs = ['point']
    if relation == 'rs-opp-sides':
        list_objs = ['point'] + ['line']
    if relation == 'rs-same-sides':
        list_objs = ['point'] + ['line']
    if relation == 'rs-closer-to-p':
        list_objs = ['point']
    if relation == 'rs-closer-to-l':
        list_objs = ['line']

    relation = render_relation(relation, list_objs, reader)
    return relation


def render_relation(relation, list_objs, reader):
    if list_objs is None:
        return ''
    
    if reader is None or reader.points is None:
        cnt_points_reader = 0
    else:
        cnt_points_reader = len(reader.points)
    if reader is None or reader.all_lines is None:
        cnt_lines_reader = 0
    else:
        cnt_lines_reader = len(reader.all_lines)
    if reader is None or reader.circles is None:
        cnt_circles_reader = 0
    else:
        cnt_circles_reader = len(reader.circles)

    cnt_points_list = 0
    cnt_lines_list = 0
    cnt_circles_list = 0
    for obj in list_objs:
        if obj == 'point':
            cnt_points_list += 1
        if obj == 'line':
            cnt_lines_list += 1
        if obj == 'circle':
            cnt_circles_list += 1 
    
    if cnt_points_list > cnt_points_reader \
        or cnt_lines_list > cnt_lines_reader \
        or cnt_circles_list > cnt_circles_reader:
        names = [(' P' + str(i + 1)) for i in range(len(list_objs))]
    else:
        names = []
        samp_points = random.sample(reader.points, cnt_points_list)
        samp_lines = random.sample(reader.all_lines, cnt_lines_list)
        samp_circles = random.sample(reader.circles, cnt_circles_list)
        for obj in list_objs:
            if obj == 'point':
                names += [' ' + samp_points.pop(0).val]
            if obj == 'lines':
                names += [' ' + samp_lines.pop(0).val]
            if obj == 'circles':
                names += [' ' + samp_circles.pop(0).val]
            if obj == 'root-selector':
                names += specify_root_selector(reader)
    
    relation = ''.join(['(', relation] + names + [')'])
    return relation

def specify_root_selector(reader):
    root_selectors = read_GMBL_list('gmbl_lang/root_selector.txt')
    root_selector = random.sample(root_selectors, 1)[0]
    if root_selector != 'rs-arbitrary':
        root_selector = specify_relation(root_selector, reader)
    return root_selector
