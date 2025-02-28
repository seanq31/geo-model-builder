import argparse
import pdb

from builder import build
from util import DEFAULTS
from graph_generator import generate_graph


if __name__ == "__main__":

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


    args = parser.parse_args()
    args = vars(args)


    # lines = open(args['problem'], 'r').readlines()
    # args['lines'] = lines

    num_steps = 10
    generate_graph(args, num_steps)
