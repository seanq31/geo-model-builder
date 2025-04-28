"""
Copyright (c) 2020 Ryan Krueger. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Ryan Krueger, Jesse Michael Han, Daniel Selsam
"""

import collections
import matplotlib.pyplot as plt
import os
import pdb
import numpy as np
import math

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Arc
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox

from adjustText import adjust_text
import matplotlib


UNNAMED_ALPHA = 0.1
MIN_AXIS_VAL = -10
MAX_AXIS_VAL = 10


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    def __init__(self, xy, p1, p2, size=55, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        t1 = self.get_theta1()
        t2 = self.get_theta2()
        if (t2 - t1) % 360 > 180:
            self.vec1 = p2
            self.vec2 = p1

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, color='blue', **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.unit(), self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s * 0.65
        # import pdb; pdb.set_trace()
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
            

class Diagram(collections.namedtuple("Diagram", ["named_points", "named_lines", "named_circles", "segments", "seg_colors", "unnamed_points", "unnamed_lines", "unnamed_circles", "ndgs", "goals", "angle_to_annotate", "segment_to_annotate"])):
    def plot(self, show=True, save=False, fname=None, return_fig=False, show_unnamed=True, show_xy_axis=False):

        unnamed_points = self.unnamed_points
        unnamed_lines = self.unnamed_lines
        unnamed_circles = self.unnamed_circles

        if not show_unnamed:
            unnamed_points = list()
            unnamed_lines = list()
            unnamed_circles = list()

        # Plot named points
        xs = [p.x for p in self.named_points.values()]
        ys = [p.y for p in self.named_points.values()]
        names = [n for n in self.named_points.keys()]

        fig, ax = plt.subplots()
        
        ax.scatter(xs, ys)
        for i, n in enumerate(names):
            ax.annotate(str(n), (xs[i], ys[i]))

        # Plot unnamed points
        u_xs = [p.x for p in unnamed_points]
        u_ys = [p.y for p in unnamed_points]
        ax.scatter(u_xs, u_ys, c="black", alpha=UNNAMED_ALPHA)

        # Plot named circles
        for (c_name, (O, r)) in self.named_circles.items():
            circle = plt.Circle((O.x, O.y),
                                radius=r,
                                fill=False,
                                label=c_name,
                                color=np.random.rand(3)
            )
            ax.add_patch(circle)

        plt.axis('scaled')
        plt.axis('square')

        # Plot lines AFTER all bounds are set

        have_points = self.named_points or unnamed_points
        have_circles = self.named_circles or unnamed_circles

        if not (have_points or have_circles):
            lo_x_lim, lo_y_lim = -2, -2
            hi_x_lim, hi_y_lim = 2, 2
            # plt.xlim(-2, 2)
            # plt.ylim(-2, 2)
        else:
            (lo_x_lim, hi_x_lim) = ax.get_xlim()
            (lo_y_lim, hi_y_lim) = ax.get_ylim()
            if self.named_lines:
                lo_x_lim -= 1
                hi_x_lim += 1
                lo_y_lim -= 1
                hi_y_lim += 1
            lo_x_lim = max(MIN_AXIS_VAL, lo_x_lim)
            hi_x_lim = min(MAX_AXIS_VAL, hi_x_lim)
            lo_y_lim = max(MIN_AXIS_VAL, lo_y_lim)
            hi_y_lim = min(MAX_AXIS_VAL, hi_y_lim)
            # ax.set_xlim([max(MIN_AXIS_VAL, lo_x_lim), min(MAX_AXIS_VAL, hi_x_lim)])
            # ax.set_ylim([max(MIN_AXIS_VAL, lo_y_lim), min(MAX_AXIS_VAL, hi_y_lim)])

        # Plot unnamed circles (always unnamed before named)
        for O, r in unnamed_circles:
            circle = plt.Circle((O.x, O.y),
                                radius=r,
                                fill=False,
                                color="black",
                                alpha=UNNAMED_ALPHA
            )
            ax.add_patch(circle)

        ax.set_xlim([lo_x_lim, hi_x_lim])
        ax.set_ylim([lo_y_lim, hi_y_lim])

        def plot_line(L, name=None):
            (nx, ny), r = L
            if nx == 0:
                l_angle = math.pi / 2
            else:
                l_angle = math.atan(ny / nx) % math.pi
            if l_angle == 0:
                if name is not None:
                    plt.axvline(x=r, label=name) # FIXME: Check if this labrel works
                else:
                    plt.axvline(x=r, c="black", alpha=UNNAMED_ALPHA)
            else:
                slope = -1 / math.tan(l_angle)
                intercept = r / math.sin(l_angle)

                eps = 0.2
                (lo_x_lim, hi_x_lim) = ax.get_xlim()
                x_vals = np.array((lo_x_lim - 0.2, hi_x_lim + 0.2))
                y_vals = intercept + slope * x_vals
                if name is not None:
                    # plt.plot(x_vals, y_vals, '--', label=l_name)
                    plt.plot(x_vals, y_vals, label=name)
                else:
                    # plt.plot(x_vals, y_vals, '--', c="black")
                    plt.plot(x_vals, y_vals, c="black", alpha=UNNAMED_ALPHA)

        # Plot unnamed lines
        for L in unnamed_lines:
            plot_line(L)

        # Plot named lines
        for l, L in self.named_lines.items():
            # ax + by = c
            l_name = l.val
            plot_line(L, l_name)

        if self.named_lines or self.named_circles:
            plt.legend()

        # Plot angle annotations
        dict_points = {ps.val: ps for ps in self.named_points}
        if self.angle_to_annotate != []:
            angles_drawn = []
            size_base = 30
            size_add = 10
            for angle in self.angle_to_annotate:
                ps = []
                stop_add = True
                for pname in angle[0]:
                    ps.append(self.named_points[dict_points[pname]])
                    stop_add = stop_add and (pname in angle[1])
                if stop_add:
                    continue
                texts = angle[1]
                # import pdb; pdb.set_trace()
                size_angle = size_base
                if angle[0][1] in angles_drawn:
                    size_angle += size_add * angles_drawn.count(angle[0][1])
                angles_drawn.append(angle[0][1])
                am1 = AngleAnnotation(ps[1], ps[0], ps[2], size=size_angle, textposition="edge", ax=ax, text=texts, color='blue')

                self.segments.append(tuple([ps[1], ps[0]]))
                self.seg_colors.append([0, 0, 0])
                self.segments.append(tuple([ps[1], ps[2]]))
                self.seg_colors.append([0, 0, 0])
        
        # Plot segment annotations
        dict_points = {ps.val: ps for ps in self.named_points}
        # import pdb; pdb.set_trace()
        if self.segment_to_annotate != []:
            for segment in self.segment_to_annotate:
                ps = []
                stop_add = True
                for pname in segment[0]:
                    stop_add = stop_add and (pname in segment[1])
                    ps.append(self.named_points[dict_points[pname]])
                if stop_add:
                    continue
                mid_ps = ps[0] + ps[1]
                mid_ps = [item/2 for item in mid_ps]

                vec_in_pixels = ps[0] - ps[1]
                if vec_in_pixels[0] != 0:
                    vec_in_pixels = [-vec_in_pixels[1]/vec_in_pixels[0], 1]
                else:
                    vec_in_pixels = [1, -vec_in_pixels[0]/vec_in_pixels[1]]
                vec_in_pixels = vec_in_pixels / np.sqrt(np.power(vec_in_pixels[0], 2) + np.power(vec_in_pixels[1], 2))
                offs = 0.1 * ax.figure.dpi / 72. 
                offs = [offs*vec_in_pixels[0], offs*vec_in_pixels[1]]

                # plt.text(mid_ps[0], mid_ps[1], f'{segment[0]}={segment[1]}', ha='center', va='bottom', color='red')
                # plt.text(mid_ps[0]+offs[0]*0.5, mid_ps[1]+offs[1]*0.5, str(segment[1]), ha='center', va='bottom', color='red')
                ax.annotate(str(segment[1]), xy=[mid_ps[0]+offs[0]*0.5, mid_ps[1]+offs[1]*0.5], xytext=[mid_ps[0]+offs[0]*0.5, mid_ps[1]+offs[1]*0.5], textcoords=ax.transData, color='red')

                ax.annotate('aaa', xy=mid_ps, xytext=mid_ps, textcoords=ax.transData, alpha=0)
                ax.annotate("", xy=[ps[0][ii]+offs[ii] for ii in range(2)], xytext=[ps[1][ii]+offs[ii] for ii in range(2)], textcoords=ax.transData, arrowprops=dict(arrowstyle='<->', color='red'))
                ax.annotate("", xy=[ps[0][ii]+offs[ii] for ii in range(2)], xytext=[ps[1][ii]+offs[ii] for ii in range(2)], textcoords=ax.transData, arrowprops=dict(arrowstyle='|-|', color='red'))
                self.segments.append(tuple([ps[1], ps[0]]))
                self.seg_colors.append([0, 0, 0])


        # Plot segments (never named)
        # import pdb; pdb.set_trace()
        seg_drawn = []
        for (p1, p2), c in zip(self.segments, self.seg_colors):
            if (p1, p2) in seg_drawn or (p2, p1) in seg_drawn:
                continue
            # plt.plot([p1.x, p2.x],[p1.y, p2.y], c=c)
            plt.plot([p1.x, p2.x],[p1.y, p2.y], c=[0,0,0])
            offs = 0.02 * ax.figure.dpi / 72. 
            # len_seg = np.sqrt(np.pow(p1[0]-p2[0],2) + np.pow(p1[1]-p2[1],2))
            # offs /= len_seg
            # ax.annotate("a", xy=[p1[ii]+(p2[ii]-p1[ii])*offs for ii in range(2)], xytext=[p2[ii]+(p1[ii]-p2[ii])*offs for ii in range(2)], textcoords=ax.transData, color='red', alpha=1)
            # ax.annotate("a", xy=[p2[ii]+(p1[ii]-p2[ii])*offs for ii in range(2)], xytext=[p1[ii]+(p2[ii]-p1[ii])*offs for ii in range(2)], textcoords=ax.transData, color='red', alpha=1)
            ax.text(p1[0]+(p2[0]-p1[0])*offs, p1[1]+(p2[1]-p1[1])*offs, "a", color='red', alpha=0, va='center', ha='center')
            ax.text(p2[0]+(p1[0]-p2[0])*offs, p2[1]+(p1[1]-p2[1])*offs, "a", color='red', alpha=0, va='center', ha='center')
            seg_drawn.append((p1, p2))

        all_texts = [child for child in ax.get_children() if isinstance(child, matplotlib.text.Text)]
        all_texts = list(filter(lambda x: x._text != '', all_texts))
        temp = adjust_text(all_texts, expand_axes=True)

        if not show_xy_axis:
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')

        if show:
            plt.show()
        if save:
            if fname is None:
                raise RuntimeError("Must supply filename if saving plot")
            # if os.path.isfile(fname):
            #     os.remove(fname)
            plt.savefig(fname)

        if return_fig:
            return [fig, ax]
