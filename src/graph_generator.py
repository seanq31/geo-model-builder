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
        'assert'
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
    
    relation = specify_relation(relation, reader)

    return relation


def get_predicate(reader):
    predicate = random.sample(gmbl_dict['predicate'], 1)[0]
    predicate = specify_relation(predicate, reader)

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
        relation = get_predicate(reader)
        line = "".join(['(', first_word, ' ', relation, ')'])

    #first_word = 'param'
    #obj_name = 'A'
    #obj_type = 'point'
    #line = "".join(['(', first_word, ' ', obj_name, ' ', obj_type, ')'])

    #print(line)
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
    lines = copy.copy(reader.problem_lines)
    cmd = parse_sexprs([line])[0]
    try:
        reader.add_cmd(cmd)
        reader.problem_lines += [line]
    except:
        reader = InstructionReader(lines)
        return [False, reader]

    return [True, reader]


def generate_graph(opts, num_steps: int, steps_to_draw: list=None, reader: InstructionReader=None, show_plot=True, save_plot=False, outf_prefix=None, encode_fig=False, max_fail=1000):
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
        num_attempts_all += 1
        if num_attempts_all > 1e4:
            print('Reached max attemps!!!')
            return [readers, figs]
        
        #print('Generate step: ' + str(cnt_steps + 1) + ', attemp: ' + str(cnt_fail + 1))
        line = generate_next_line(reader)
        #print('generated line: ' + line)
        line_is_feasible, reader = add_new_line(reader, line)
        
        # check if feasible
        if line_is_feasible:
            cnt_steps += 1
            cnt_fail = 0
            lines_fail = []
            if cnt_steps in steps_to_draw:
                try:
                    reader = InstructionReader(reader.problem_lines)
                    figs += [solve_draw(opts, reader, show_plot, save_plot, outf_prefix, encode_fig)]
                    readers += [reader]
                    print('######################## Lines drawn above ########################')
                    print(reader.problem_lines)
                    print('')
                    cp_lines = copy.copy(reader.problem_lines)
                except:
                    cnt_fail = 0
                    if cp_lines is None:
                        reader = None
                        cnt_steps = 0
                    else:
                        reader = InstructionReader(cp_lines)
                        cnt_steps = len(cp_lines)

        else:
            cnt_fail += 1
            lines_fail += [line]

    return [readers, figs]


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
            figs.append(m.plot(show=show_plot, save=save_plot, fname=f"{outf_prefix}_{i}.png", return_fig=encode_fig, show_unnamed=opts['unnamed_objects']))
    
    return figs


