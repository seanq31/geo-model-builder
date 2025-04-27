"""
Copyright (c) 2020 Ryan Krueger. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Ryan Krueger, Jesse Michael Han, Daniel Selsam
"""

import collections
import pdb
import random
import string


class Root(collections.namedtuple("Root", ["pred", "vars"])):
    def __str__(self):
        if self.pred == "arbitrary":
            return "root-arbitrary"
        else:
            return f"(root-{self.pred} {' '.join([str(v) for v in self.vars])})"


Bucket = collections.namedtuple("Bucket", ["points", "assertions"])

FuncInfo = collections.namedtuple("FuncInfo", ["head", "args"])

def is_sample_pred(pred):
    return pred in ["triangle", "polygon"]

def group_pairs(p, ps):
    if len(ps) != 4:
        raise RuntimeError("[group_pairs] Wrong number of points passed to group_pairs")
    a, b, c, d = ps
    if p == a and p not in [b, c, d]:
        return (b, (c, d))
    elif p == b and p not in [a, c, d]:
        return (a, (c, d))
    elif p == c and p not in [a, b, d]:
        return (d, (a, b))
    elif p == d and p not in [a, b, c]:
        return (c, (a, b))
    return (None, None)

def match_in_first_2(p, ps):
    if len(ps) != 4:
        raise RuntimeError("[match_in_first_2] Wrong number of points passed to match_in_first_2")
    x, y, a, b = ps
    if p == x and p not in [y, a, b]:
        return True, (y, a, b)
    if p == y and p not in [x, a, b]:
        return True, (x, a, b)
    return (False, None)

DEFAULTS = {
    "decay_steps": 1e3,
    "decay_rate": 0.7,
    "distinct_prob": 1.0, # Note this
    "eps": 1e-3,
    "enforce_goals": False,
    "experiment": False,
    "learning_rate": 1e-1,
    "loss_freq": 100,
    "losses_freq": 1000,
    "make_distinct": 1e-2,
    "min_dist": 0.1,
    "n_iterations": 5000,
    "ndg_loss": 1e-3,
    "plot_freq": 1000,
    "unnamed_objects": True,
    "regularize_points": 1e-6,
    "n_models": 1,
    "n_tries": 3,
    "n_inits": 10,
    "verbosity": 0
}

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


"""
CASE_FIX = {
    "acuteisotri": "acuteIsoTri",
    "acutetri": "acuteTri",
    "add": "add",
    "amidpopp": "amidpOpp",
    "amidpsame": "amidpSame",
    "area": "area",
    "between": "between",
    "centroid": "centroid",
    "rsclosertop": "closerToP",
    "rsclosertol": "closerToL",
    "circ": "circ",
    "circle": "circle",
    "circumcenter": "circumcenter",
    "circumcircle": "circumcircle",
    "coa": "coa",
    "coll": "coll",
    "coords": "coords",
    "concur": "concur",
    "cong": "cong",
    "cycl": "cycl",
    "diam": "diam",
    "dist": "dist",
    "div": "div",
    "ebisector": "ebisector",
    "equitri": "equiTri",
    "eq": "eq",
    "eqn": "eqN",
    "eqp": "eqP",
    "eqangle": "eqangle",
    "eqratio": "eqratio",
    "excenter": "excenter",
    "excircle": "excircle",
    "foot": "foot",
    "gt": "gt",
    "gte": "gte",
    "harmonicconj": "harmonicConj",
    "ibisector": "ibisector",
    "incenter": "incenter",
    "incircle": "incircle",
    "inpoly": "inPoly",
    "intercc": "interCC",
    "interlc": "interLC",
    "interll": "interLL",
    "isogonal": "isogonal",
    "isogonalconj": "isogonalConj",
    "isotomic": "isotomic",
    "isotomicconj": "isotomicConj",
    "isotri": "isoTri",
    "line": "line",
    "lt": "lt",
    "lte": "lte",
    "midp": "midp",
    "midpfrom": "midpFrom",
    "mixtilinearincircle": "mixtilinearIncircle",
    "mixtilinearincenter": "mixtilinearIncenter",
    "mul": "mul",
    "neg": "neg",
    "oncirc": "onCirc",
    "online": "onLine",
    "onray": "onRay",
    "onrayopp": "onRayOpp",
    "onseg": "onSeg",
    "origin": "origin",
    "orthocenter": "orthocenter",
    "para": "para",
    "paraat": "paraAt",
    "perp": "perp",
    "perpat": "perpAt",
    "perpbis": "perpBis",
    "polygon": "polygon",
    "pow": "pow",
    "radius": "radius",
    "reflectpl": "reflectPL",
    "reflectll": "reflectLL",
    "right": "right",
    "righttri": "rightTri",
    "rsneq": "rsNeq",
    "rsoppsides": "rsOppSides",
    "sameside": "sameSide",
    "oppsides": "oppSides",
    "sqrt": "sqrt",
    "sub": "sub",
    "tangentcc": "tangentCC",
    "tangentcl": "tangentCL",
    "tangentlc": "tangentLC",
    "tangentatcc": "tangentAtCC",
    "tangentatlc": "tangentAtLC",
    "throughc": "throughC",
    "throughl": "throughL",
    "triangle": "triangle",
    "uangle": "uangle"
}
"""
