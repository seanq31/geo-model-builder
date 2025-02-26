import random

def specify_relation(first_word, obj_type, relation, reader):


    return relation


def get_name_special_param(obj_type, reader):
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
        obj_type = ''.join('(', obj_type, ' ', samp, ')')

    return [obj_name, obj_type]


def specify_predicate(predicate, reader):
    cnt_points_reader = len(reader.points)
    cnt_lines_reader = len(reader.all_lines)
    cnt_circles_reader = len(reader.circles)

    if predicate == 'centroid':
        list_objs = ['point'] * 4
    elif predicate == 'concur':
        list_objs = ['line'] * 3
    elif predicate == 'circumcenter':
        list_objs = ['point'] * 4
    elif predicate == 'cong':
        list_objs = ['point'] * 4
    elif predicate == 'contri':
        list_objs = ['point'] * 6
    elif predicate == 'coll':
        list_objs = ['point'] * 3
    elif predicate == 'cycl':
        num = random.sample(range(4, cnt_points_reader + 1), 1)
        list_objs = ['point'] * num
    elif predicate == '=':
        rand_num = random.uniform(0, 1)
        if cnt_points_reader >= 2 and rand_num < 0.7:
            list_objs = ['point'] * 2
        elif cnt_lines_reader >= 2:
            list_objs = ['line'] * 2
    elif predicate == 'eq-ratio':
        list_objs = ['point'] * 8
    elif predicate == 'foot':
        list_objs = ['point'] * 2 + ['line']   
    elif predicate == 'incenter':
        list_objs = ['point'] * 4
    elif predicate == 'inter-ll':
        list_objs = ['point'] + ['line'] * 2
    elif predicate == 'midp':
        list_objs = ['point'] * 3
    elif predicate == 'on-circ':
        list_objs = ['point'] + ['circle']
    elif predicate == 'on-line':
        list_objs = ['point'] + ['line']
    elif predicate == 'on-ray':
        list_objs = ['point'] * 3
    elif predicate == 'on-seg':
        list_objs = ['point'] * 3
    elif predicate == 'opp-sides':
        list_objs = ['point'] * 2 + ['line']
    elif predicate == 'orthocenter':
        list_objs = ['point'] * 4
    elif predicate == 'perp':
        list_objs = ['line'] * 2
    elif predicate == 'para':
        list_objs = ['line'] * 2
    elif predicate == 'same-side':
        list_objs = ['point'] * 2 + ['line']
    elif predicate == 'sim-tri':
        list_objs = ['point'] * 6
    elif predicate == 'tangent-cc':
        list_objs = ['circle'] * 2
    elif predicate == 'tangent-lc':
        list_objs = ['line'] + ['circle']
    elif predicate == 'tangent-at-cc':
        list_objs = ['point'] + ['circle'] * 2
    elif predicate == 'tangent-at-lc':
        list_objs = ['point'] + ['line'] + ['circle']

    predicate = render_predicate(predicate, list_objs, reader)
    return predicate


def render_predicate(predicate, list_objs, reader):
    cnt_points_reader = len(reader.points)
    cnt_lines_reader = len(reader.all_lines)
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
        names = [' P' + str(i + 1) for i in range(sum(list_objs))]
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
    
    predicate = ''.join(['(', predicate] + names + [')'])
    return predicate