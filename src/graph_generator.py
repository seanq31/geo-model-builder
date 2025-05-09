from instruction_reader import InstructionReader
from parse import parse_sexprs
from util_graph_gen import *

import tensorflow.compat.v1 as tf
import os
import pdb
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import copy
import math

from tf_optimizer import TfOptimizer
from parse import parse_sexprs
from instruction_reader import InstructionReader


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_first_word(reader):    
    if reader is None or (reader.points == [] and reader.lines == [] and reader.circles == []):
        return 'param'
    
    first_word_list = [
        'param',
        'define',
        'assert',
    ]
    first_word = random.sample(first_word_list, 1)[0]

    return first_word


def get_obj_type(first_word):
    if first_word == 'define':
        #obj_type = random.sample(['point', 'line', 'circle', 'number'], 1)[0]
        obj_type = random.sample(['point', 'line', 'circle'], 1)[0]
    else:
        obj_type = random.sample(gmbl_dict['param'], 1)[0]

    return obj_type


def get_obj_name(obj_type, reader):
    if obj_type == 'point':
        if reader is None or reader.points is None:
            cnt_points_reader = 0
        else:
            cnt_points_reader = len(reader.points)
        obj_name = 'P' + str(cnt_points_reader + 1)
    elif obj_type == 'line':
        if reader is None or reader.lines is None:
            cnt_lines_reader = 0
        else:
            cnt_lines_reader = len(reader.lines)
        obj_name = 'L' + str(cnt_lines_reader + 1)
    elif obj_type == 'circle':
        if reader is None or reader.circles is None:
            cnt_circles_reader = 0
        else:
            cnt_circles_reader = len(reader.circles)
        obj_name = 'C' + str(cnt_circles_reader + 1)
    elif obj_type == 'number':
        obj_name = str(round(random.normalvariate(0, 5)))
    else:
        obj_name, obj_type = get_name_special_param(obj_type, reader)

    return [obj_name, obj_type]


def get_relation(first_word, obj_type, reader):
    if first_word == 'define':
        if obj_type == 'point':
            relation = random.sample(gmbl_dict['functions_point'], 1)[0]
        elif obj_type == 'line':
            relation = random.sample(gmbl_dict['functions_line'], 1)[0]
        elif obj_type == 'circle':
            relation = random.sample(gmbl_dict['functions_circle'], 1)[0]
        elif obj_type == 'number':
            relation = random.sample(gmbl_dict['functions_number'], 1)[0]
    else:
        if obj_type == 'point':
            relation = random.sample(gmbl_dict['parameterization_point'], 1)[0]
        elif obj_type == 'line':
            relation = random.sample(gmbl_dict['parameterization_line'], 1)[0]
        elif obj_type == 'circle':
            relation = random.sample(gmbl_dict['parameterization_circle'], 1)[0]
        else:
            relation = 'any'
    
    relation = specify_relation(first_word, relation, reader)

    return relation


def get_predicate(first_word, reader):
    predicate = random.sample(gmbl_dict['predicate'], 1)[0]
    predicate = specify_relation(first_word, predicate, reader)

    return predicate


def generate_next_line(reader):
    # randomly select one line
    first_word = get_first_word(reader)
    #print(first_word)
    if first_word in ['define', 'param']:
        obj_type = get_obj_type(first_word)
        #print(obj_type)
        relation = get_relation(first_word, obj_type, reader)
        #print(relation)
        obj_name, obj_type = get_obj_name(obj_type, reader)
        #print([obj_name, obj_type])

        if relation is not None:
            line = "".join(['(', first_word, ' ', obj_name, ' ', obj_type, ' ', relation, ')'])
        else:
            line = "".join(['(', first_word, ' ', obj_name, ' ', obj_type, ')'])
        
    else:
        relation = get_predicate(first_word, reader)
        line = "".join(['(', first_word, ' ', relation, ')'])

    #first_word = 'param'
    #obj_name = 'A'
    #obj_type = 'point'
    #line = "".join(['(', first_word, ' ', obj_name, ' ', obj_type, ')'])

    #print(line)

    if first_word == 'param' and obj_type == 'point' and relation is None:
        line = 'skip this line'
    if first_word == 'param' and obj_type == 'circle' and relation is None:
        line = 'skip this line'

    if first_word == 'param' and obj_type in ['trapezoid', 'parallelogram', 'rectangle', 'square', 'diamond']:
        if reader is None or reader.points is None:
            cnt = 0
        else:
            cnt = len(reader.points)
        
        line = "".join(['(param ', obj_name, ' polygon)'])
        temp_lines = [line]
        # temp_lines = []
        # for ii in range(4):
        #     line = ''.join(['(param P', str(cnt + 1 + ii), ' point)'])
        #     temp_lines += [line]
        temp_lines += ['(assert (para (line P' + str(cnt + 1) + ' P' + str(cnt + 2) + ') (line P' + str(cnt + 3) + ' P' + str(cnt + 4) + ')))']

        if obj_type in ['parallelogram', 'rectangle', 'square', 'diamond']:
            temp_lines += ['(assert (para (line P' + str(cnt + 1) + ' P' + str(cnt + 4) + ') (line P' + str(cnt + 2) + ' P' + str(cnt + 3) + ')))']

        if obj_type in ['rectangle', 'square']:
            temp_lines += ['(assert (perp (line P' + str(cnt + 1) + ' P' + str(cnt + 2) + ') (line P' + str(cnt + 1) + ' P' + str(cnt + 4) + ')))']
        
        if obj_type in ['square', 'diamond']:
            temp_lines += ['(assert (cong P' + str(cnt + 1) + ' P' + str(cnt + 2) + ' P' + str(cnt + 1) + ' P' + str(cnt + 4) + '))']

        return temp_lines

    return [line]


def generate_next_eval(first_word, reader):
    relation = get_predicate(first_word, reader)
    line = "".join(['(eval ', relation, ')'])

    return line


def add_new_line(reader: InstructionReader, line: str):
    # try creating the first object
    if reader is None:
        try:
            reader = InstructionReader([line])
        except:
            return [False, None]
        
        return [True, reader]

    # try adding the new line
    lines = copy.deepcopy(reader)
    cmd = parse_sexprs([line])[0]
    try:
        reader.add_cmd(cmd)
        reader.problem_lines += [line]
    except:
        reader = InstructionReader(lines.problem_lines)
        return [False, reader]

    return [True, reader]


def generate_graph(opts, num_steps: int, num_eval: int=1, steps_to_draw: list=None, reader: InstructionReader=None, show_plot=True, save_plot=False, outf_prefix=None, encode_fig=False, max_fail=1000, max_eval_attempt=100):
    if steps_to_draw is None:
        steps_to_draw = [1, (num_steps + 1) // 2, num_steps]

    # get the GMBL language bank
    global gmbl_dict
    gmbl_dict = {}
    file_list = os.listdir('gmbl_lang')
    for filename in file_list:
        gmbl_dict[filename[: -4]] = read_GMBL_list('gmbl_lang/' + filename)
    
    # generate a few graphs within number of steps (from scarth or a given reader)
    cnt_steps = 0
    cnt_fail = 0
    num_attempts_all = 0
    cp_lines = None
    lines_fail = []
    readers = []
    figs = []
    while cnt_steps < num_steps and cnt_fail < max_fail:
        #print('Generate step: ' + str(cnt_steps + 1) + ', attemp: ' + str(cnt_fail + 1))
        lines = generate_next_line(reader)
        #print('generated line: ' + line)
        line_is_feasible = True
        for line in lines:
            temp_feasible, reader = add_new_line(reader, line)
            line_is_feasible = line_is_feasible and temp_feasible
        
        # check if feasible
        if line_is_feasible:
            cnt_steps += 1
            cnt_fail = 0
            lines_fail = []
            if cnt_steps in steps_to_draw:
                try:
                    reader = InstructionReader(reader.problem_lines)
                    num_attempts_all += 1
                    fig = solve_draw(opts, reader, show_plot, save_plot, outf_prefix, encode_fig)            
                    figs += [fig]
                    readers += [reader]
                    print('######################## Lines drawn above ########################')
                    print(reader.problem_lines)
                    print('')
                    cp_lines = copy.deepcopy(reader)
                except:
                    cnt_fail = 0
                    if cp_lines is None:
                        reader = None
                        cnt_steps = 0
                    else:
                        reader = InstructionReader(cp_lines.problem_lines)
                        temp_draw = [i for i, x in enumerate(steps_to_draw) if x == cnt_steps][0]
                        if temp_draw == 0:
                            cnt_steps = 0
                        else:
                            cnt_steps = steps_to_draw[temp_draw - 1]
                        cp_lines = copy.deepcopy(reader)
        else:
            cnt_fail += 1
            lines_fail += [line]
        
        if num_attempts_all > 1e4:
            print('Reached max attemps!!!')
            return [readers, figs]
    
    # get evaluation results
    pr_reader_num = len(readers)
    cnt_eval = 0
    cnt_eval_attempt = 0
    while cnt_eval < num_eval and cnt_eval_attempt < max_eval_attempt:
        cnt_eval_attempt += 1
        reader, fig = generate_eval(opts, readers[pr_reader_num - 1], max_eval_attempt=max_eval_attempt)
        if reader is not None:
            cnt_eval += 1
            readers += [reader]
            figs += [fig]
    
    return [readers, figs]


def generate_eval(opts, reader, show_plot=True, save_plot=False, outf_prefix=None, encode_fig=False, max_eval_attempt=100):
    cnt_pr_lines = len(reader.problem_lines)
    cnt_eval = 0
    cnt_eval_attempt = 0
    num_eval = 1
    while cnt_eval < num_eval and cnt_eval_attempt < max_eval_attempt:
        line_eval = generate_next_eval('eval', reader)
        eval_is_feasible, reader = add_new_line(reader, line_eval)
        if eval_is_feasible:
            try:
                reader = InstructionReader(reader.problem_lines)
            except:
                reader = InstructionReader(reader.problem_lines[:cnt_pr_lines])
                continue
            
            cnt_eval_attempt += 1
            fig = []
            try:
                fig = solve_draw(opts, reader, show_plot, save_plot, outf_prefix, encode_fig)
            except:
                reader = InstructionReader(reader.problem_lines[:cnt_pr_lines])
            finally:
                if fig != []:
                    cnt_eval += 1                    
                    print('######################## Lines drawn above ########################')
                    print(reader.problem_lines)
                    print('')

        if cnt_eval_attempt >= max_eval_attempt:
            print('Reach max eval attempts!')
            return [None, None]

    return [reader, fig]


def solve_draw(opts, reader, show_plot=True, save_plot=False, outf_prefix=None, encode_fig=False):
    instructions = reader.instructions

    verbosity = opts['verbosity']

    if verbosity >= 0:
        print("INPUT INSTRUCTIONS:\n{instrs_str}".format(instrs_str="\n".join([str(i) for i in instructions])))

    g = tf.Graph()
    with g.as_default():

        solver = TfOptimizer(instructions, opts,
                             reader.unnamed_points, reader.unnamed_lines, reader.unnamed_circles,
                             reader.segments, reader.seg_colors, g)
        solver.preprocess()
        filtered_models = solver.solve()
        # print(filtered_models)


    if verbosity >= 0:
        print(f"\n\nFound {len(filtered_models)} models")

    figs = list()
    for i, m in enumerate(filtered_models):
        # FIXME: Inconsistent return type
        if not (encode_fig or show_plot or save_plot):
            figs.append(m)
        else:
            m.plot(show=show_plot, save=save_plot, fname=f"{outf_prefix}_{i}.png", return_fig=encode_fig, show_unnamed=opts['unnamed_objects'])
            figs.append(m)
    
    if figs == []:
        raise RuntimeError(f"Fail to solve this graph!")
    
    return figs


