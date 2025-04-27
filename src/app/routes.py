"""
Copyright (c) 2020 Ryan Krueger. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Ryan Krueger, Jesse Michael Han, Daniel Selsam
"""

from flask import render_template, request, Response, send_file, jsonify
from app import app
from io import StringIO, BytesIO
import pdb
import base64

from builder import build
from util import DEFAULTS

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    try:
        jsdata = request.form['problem_input']
        lines = str(jsdata).split('\n')

        args = DEFAULTS
        args['lines'] = lines
        args['n_models'] = int(request.form['n_models'])
        args['plot_freq'] = -1
        args['losses_freq'] = -1
        args['loss_freq'] = -1
        args['unnamed_objects'] = (request.form['plot_unnamed'] == 'true')

        figs = build(args, show_plot=False, encode_fig=True)
        urls = list()

        for fig in figs:
            img = BytesIO()
            fig.savefig(img, format='png')
            fig.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            urls.append(f"data:image/png;base64,{plot_url}")

        return jsonify(srcs=urls)
        # return f"data:image/png;base64,{plot_url}"

    except Exception as e:
        return Response(
            str(e),
            status=400
        )
