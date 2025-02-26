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
        'assert'
    ]
    first_word = random.sample(first_word_list, 1)

    return first_word


def get_obj_type(first_word):
    if first_word == 'define':
        obj_type = random.sample(['point', 'line', 'circle', 'number'])
    else:
        obj_type = random.sample(gmbl_dict['param'])

    return obj_type


def get_obj_name(obj_type, reader):
    if obj_type == 'point':
        cnt = len(reader.points)
        obj_name = 'P' + str(cnt + 1)
    elif obj_type == 'line':
        cnt = len(reader.lines)
        obj_name = 'L' + str(cnt + 1)
    elif obj_type == 'circle':
        cnt = len(reader.circles)
        obj_name = 'C' + str(cnt + 1)
    elif obj_type == 'number':
        obj_name = str(round(random.normalvariate(0, 5)))
    else:
        obj_name, obj_type = get_name_special_param(obj_type, reader)

    return [obj_name, obj_type]


def get_relation(first_word, obj_type, reader):
    if first_word == 'define':
        if obj_type == 'point':
            relation = random.sample(gmbl_dict['functions_point'])
        elif obj_type == 'line':
            relation = random.sample(gmbl_dict['functions_line'])
        elif obj_type == 'circle':
            relation = random.sample(gmbl_dict['functions_circle'])
        elif obj_type == 'number':
            relation = random.sample(gmbl_dict['functions_number'])
    else:
        if obj_type == 'point':
            relation = random.sample(gmbl_dict['parameterization_point'])
        elif obj_type == 'line':
            relation = random.sample(gmbl_dict['parameterization_line'])
        elif obj_type == 'circle':
            relation = random.sample(gmbl_dict['parameterization_circle'])
        else:
            relation = 'any'

    relation = specify_relation(first_word, obj_type, relation, reader)
    return relation


def get_predicate(reader):
    predicate = random.sample(gmbl_dict['predicate'], 1)
    predicate = specify_predicate(predicate, reader)

    return predicate


def generate_next_line(reader):
    # randomly select one line
    first_word = get_first_word(reader)
    if first_word in ['define', 'param']:
        obj_type = get_obj_type(first_word)
        relation = get_relation(first_word, obj_type, reader)
        obj_name, obj_type = get_obj_name(obj_type, relation, reader)
    else:
        relation = get_predicate(reader)
    

    if relation is not None:
        relation = ' '.join([obj_type, relation])

    first_word = 'param'
    obj_name = 'A'
    relation = 'point'
    line = "".join(['(', first_word, ' ', obj_name, ' ', obj_type, ')'])
    
    return line


def add_new_line(prev_reader: InstructionReader, line: str):
    # try creating the first object
    if prev_reader is None:
        try:
            new_reader = InstructionReader([line])
            return [True, new_reader]
        except:
            return [False, prev_reader]

    # try adding the new line
    new_reader = copy.copy(prev_reader)
    cmd = parse_sexprs([line])[0]
    line_is_feasible = new_reader.add_cmd(cmd)

    if line_is_feasible:
        return [True, new_reader]
    else:
        return [False, prev_reader]
    

def read_GMBL_list(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    return lines
    

def generate_graph(num_steps: int, opts, reader: InstructionReader=None, show_plot=True, save_plot=False, outf_prefix=None, encode_fig=False, max_fail=1000):
    # get the GMBL language bank
    global gmbl_dict
    gmbl_dict = {}
    file_list = os.listdir('gmbl_lang')
    for filename in file_list:
        gmbl_dict[filename[: -4]] = read_GMBL_list('gmbl_lang/' + filename)
    
    # generate a few graphs within number of steps (from scarth or a given reader)
    cnt_steps = 0
    cnt_fail = 0
    lines_fail = []
    while cnt_steps < num_steps and cnt_fail < max_fail:
        line = generate_next_line(reader)
        line_is_feasible, reader = add_new_line(reader, line)
        
        # check if feasible
        if line_is_feasible:
            cnt_steps += 1
            cnt_fail = 0
            lines_fail = []
            solve_draw(num_steps, cnt_steps, opts, reader, show_plot, save_plot, outf_prefix, encode_fig)
        else:
            cnt_fail += 1
            lines_fail += [line]

    return


def solve_draw(num_steps, cnt_steps, opts, reader, show_plot, save_plot, outf_prefix, encode_fig):
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
    
    print(reader.problem_lines)
    #print(lines_fail)
    if cnt_steps < num_steps:
        raise Warning(f"Fail to generate at step {cnt_steps + 1}")
    return figs


