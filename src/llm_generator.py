from openai import OpenAI
import re
import json

import argparse
import pdb
from builder import build
from util import DEFAULTS

from multiprocessing import Pool
import pickle
import datetime

# Parse arguments
parser = argparse.ArgumentParser(description='Arguments for building a model that satisfies a set of geometry constraints')

# General arguments
parser.add_argument('--problem', '-p', action='store', type=str, help='Name of the file defining the set of constraints')
parser.add_argument('--dir', '-d', action='store', type=str, help='Directory containing problem files.')
parser.add_argument('--regularize_points', action='store', dest='regularize_points', type=float, default=DEFAULTS["regularize_points"])
parser.add_argument('--make_distinct', action='store', dest='make_distinct', type=float, default=DEFAULTS["make_distinct"])
parser.add_argument('--distinct_prob', action='store', dest='distinct_prob', type=float, default=DEFAULTS["distinct_prob"])
parser.add_argument('--min_dist', action='store', dest='min_dist', type=float, default=DEFAULTS["min_dist"])
parser.add_argument('--ndg_loss', action='store', dest='ndg_loss', type=float, default=DEFAULTS["ndg_loss"])

parser.add_argument('--n_models', action='store', dest='n_models', type=int, default=DEFAULTS['n_models'])
parser.add_argument('--n_tries', action='store', dest='n_tries', type=int, default=DEFAULTS['n_tries'])
parser.add_argument('--n_inits', action='store', dest='n_inits', type=int, default=DEFAULTS['n_inits'])
parser.add_argument('--verbosity', action='store', dest='verbosity', type=int, default=DEFAULTS['verbosity'])
parser.add_argument('--enforce_goals', dest='enforce_goals', action='store_true')
parser.add_argument('--plot_freq', action='store', dest='plot_freq', type=int, default=DEFAULTS['plot_freq'])
parser.add_argument('--loss_freq', action='store', dest='loss_freq', type=int, default=DEFAULTS['loss_freq'])
parser.add_argument('--losses_freq', action='store', dest='losses_freq', type=int, default=DEFAULTS['losses_freq'])

parser.add_argument('--unnamed_objects', dest='unnamed_objects', action='store_true')
parser.add_argument('--no_unnamed_objects', dest='unnamed_objects', action='store_false')
parser.set_defaults(unnamed_objects=True)

# Tensorflow arguments
parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=DEFAULTS["learning_rate"])
parser.add_argument('--decay_steps', action='store', dest='decay_steps', type=float, default=DEFAULTS["decay_steps"])
parser.add_argument('--decay_rate', action='store', dest='decay_rate', type=float, default=DEFAULTS["decay_rate"])
parser.add_argument('--n_iterations', action='store', dest='n_iterations', type=int, default=DEFAULTS["n_iterations"])
parser.add_argument('--eps', action='store', dest='eps', type=float, default=DEFAULTS["eps"])

parser.add_argument('--experiment', dest='experiment', action='store_true')

parser.add_argument("--f", type=str, default='abc.def')

global args
args = parser.parse_args()
args = vars(args)

args['n_tries'] = 10
args['eps'] = 5e-5
args['n_iterations'] = 10000
args['verbosity'] = -2


def llm_generate(messages):
    messages, input_caption = messages
    client = OpenAI(
        api_key = "c8a7e43e-cce1-4d61-af75-de350b7e5731",
        base_url = "https://ark.cn-beijing.volces.com/api/v3",
    )

    print("----- standard request -----")
    # print(messages[0]['content'])
    print(input_caption)

    completion = client.chat.completions.create(
        model = "deepseek-r1-250120", 
        messages = messages,
    )

    # print("----- Reasoning content -----")
    # print(completion.choices[0].message.reasoning_content)

    # print("----- Content -----")
    # print(completion.choices[0].message.content)

    return([messages, completion, input_caption])


def gmb_draw(inputs):
    content, reasoning_content, input_caption = inputs
    try:
        re_find = re.finditer('```', content)
        inds = []
        for ind in re_find:
            inds += [ind.span()]

        lines = content[inds[0][1] : inds[1][0]]
        lines = lines.splitlines()[1:]

        print(lines)
    except:
        try:
            re_find = re.finditer('```', reasoning_content)
            inds = []
            for ind in re_find:
                inds += [ind.span()]

            lines = reasoning_content[inds[0][1] : inds[1][0]]
            lines = lines.splitlines()[1:]

            print(lines)
        except:
            raise RuntimeError("No code block found in the reasoning content or content")
        
    args['lines'] = lines

    try:
        res = build(args)
    except:
        res = [lines, [None]]

    try:
        if res[1] != [] and res[1][0] is not None:
            segments_list = get_segments(content, reasoning_content)
            dict_points = {ps.val: ps for ps in res[1][0].named_points}
            for seg in segments_list:
                ps_seg = tuple([res[1][0].named_points[dict_points[ps]] for ps in seg])
                res[1][0].segments.append(ps_seg)
                res[1][0].seg_colors.append([0, 0, 0])
        else:
            pass
    except:
        print("Fail to add additional segments!!!")

    return res + [input_caption]


def get_segments(content, reasoning_content):
    try:
        re_find = re.finditer('```', content)
        inds = []
        for ind in re_find:
            inds += [ind.span()]

        re_find = re.finditer(';; segments to connect: ', content)
        inds_seg = []
        for ind in re_find:
            inds_seg += [ind.span()]

        lines = content[inds_seg[0][1] : inds[1][0]]
        lines = lines.upper().splitlines()[0].split(' ')
    except:
        try:
            re_find = re.finditer('```', reasoning_content)
            inds = []
            for ind in re_find:
                inds += [ind.span()]

            re_find = re.finditer(';; segments to connect: ', reasoning_content)
            inds_seg = []
            for ind in re_find:
                inds_seg += [ind.span()]

            lines = reasoning_content[inds_seg[0][1] : inds[1][0]]
            lines = lines.upper().splitlines()[0].split(' ')
        except:
            raise RuntimeError("No segment list found in the reasoning content or content")
    
    if lines is None or lines == []:
        return []

    segments_list = []
    for segment in lines:
        try:
            assert(len(segment) == 2)
            segments_list += [[segment[0], segment[1]]]
        except:
            print("Fail to get segment " + segment + " !!!")
    
    return segments_list


def try_generate_and_draw(messages):
    num_try_gen = 3
    num_try_draw = 2
    cnt_try_gen = 0

    while cnt_try_gen < num_try_gen:
        cnt_try_gen += 1
        cnt_try_draw = 0

        res_temp = llm_generate(messages)
        contents_and_reasoning = [res_temp[1].choices[0].message.content, res_temp[1].choices[0].message.reasoning_content, res_temp[2]]
        while cnt_try_draw < num_try_draw:
            cnt_try_draw += 1
            res = gmb_draw(contents_and_reasoning)
            if res[1] == [] or res[1][0] is None:
                continue
            else:
                return res + contents_and_reasoning
        
        args['eps'] *= 5
        res = gmb_draw(contents_and_reasoning)
        args['eps'] /= 5
        if res[1] == [] or res[1][0] is None:
            continue
        else:
            return res + contents_and_reasoning
    
    return res + contents_and_reasoning


if __name__ == '__main__':
    time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    cnt = -1
    input_captions = []
    with open("C:\qinshenghao\桌面\geoqa_v2_99k_20250103.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            cnt += 1
            data = json.loads(line)
            if cnt == 29 * 32:
                break
            if cnt % 29 != 0:
                continue

            content = data['conversations'][1]['value']
            #print(data['conversations'][0]['value'])
            #print(data['conversations'][1]['value'])
            # inds = []
            # re_find = re.finditer('Information in the Image', content)
            # for ind in re_find:
            #     inds += [ind.span()]
            # re_find = re.finditer('Terms and Knowledge Points', content)
            # for ind in re_find:
            #     inds += [ind.span()]

            # inds = []
            # re_find = re.finditer('<image>\n', content)
            # for ind in re_find:
            #     inds += [ind.span()]
            # re_find = re.finditer('\nA', content)
            # for ind in re_find:
            #     inds += [ind.span()]

            
            inds = []
            re_find = re.finditer('## 列出已知信息', content)
            for ind in re_find:
                inds += [ind.span()]
            re_find = re.finditer('## 术语与知识点', content)
            for ind in re_find:
                inds += [ind.span()]

            lines = content[inds[0][1] : inds[1][0]]
            lines = ''.join(lines.splitlines()[:-1])
            lines = ''.join(['\n', lines, '\n'])

            input_captions += [lines]

    filename = '..\grammar_example.txt'
    with open(filename, 'r', encoding='utf-8') as file:
        grammer_example = [line for line in file]
    grammer_example = ''.join(grammer_example)

    quest1 = '根据文档中的GMBL的语法和示例，使用GMBL语法，生成如下几何题对应的代码。'
    quest2 = '请确保每个点在使用前都已经定义过，确保没有重复定义点或者图形，确保符合示例的GMBL语法，确保每一句语句都写在同一行，确保GMBL输入的角度是弧度制，确保没有形如(define XXX number xxx)或(param XXX number xxx)的语句，确保语句中没有使用and或者or。'
    quest3 = '把最终回答放在代码block内。最后列出所有需要连接的线段，以";; segments to connect: AB AC BC"的形式罗列。'

    messages_list = [
        [[{
        "role": "user", 
        "content": grammer_example + '\n\n' + quest1 + input_caption + quest2 + quest3
        }], input_caption]
        for input_caption in input_captions]

    # with Pool(8) as p:
    #     res = p.map(llm_generate, messages_list)
    # llm_result_dict = {i: result for i, result in enumerate(res)}
    # time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # file_name = '../llm_gen/res_llm_gen' + time_str + '.pkl'
    # with open(file_name, 'wb') as f:
    #     pickle.dump(llm_result_dict, f)

    # # with open('../llm_gen/res_llm_gen.pkl', 'rb') as f:
    # #     llm_result_dict = pickle.load(f)

    # all_contents_and_reasoning= [[res[1][1].choices[0].message.content, res[1][1].choices[0].message.reasoning_content, res[1][2]] for res in llm_result_dict.items()]
    # with Pool(8) as p:
    #     res = p.map(gmb_draw, all_contents_and_reasoning)

    # # res = []
    # # with Pool(5) as p:
    # #     for _ in range(1):
    # #         res_temp = p.map(gmb_draw, all_contents_and_reasoning)

    # #     if res == []:
    # #         res = res_temp
    # #     else:
    # #         for i, res_t in enumerate(res_temp):
    # #             if res_t[1] != [] and res_t[1][0] is not None:
    # #                 res[i][1][0] = res_t[1][0]

    # gmb_result_dict = {i: result for i, result in enumerate(res)}
    # with open('../llm_gen/res_gmb_draw.pkl', 'wb') as f:
    #     pickle.dump(gmb_result_dict, f)

    with Pool(8) as p:
        res = p.map(try_generate_and_draw, messages_list)
    res_all_in_one_dict = {i: result for i, result in enumerate(res)}

    file_name = '../llm_gen/res_all_in_one_dict_' + time_str + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(res_all_in_one_dict, f)