# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection


#%%---------------------------------------------------------------------------#
def scale_radar(array, n=6, kind='max'):
    
    from matplotlib.ticker import MaxNLocator, LinearLocator
    
    
    if array.ndim == 1:
        
        array = np.atleast_2d(array).T
        
    if kind == 'linear':
        locator = LinearLocator(n)
    else:
        locator = MaxNLocator(n)
        
    tvalues = ([locator.tick_values(*aminmax) for aminmax
                        in zip(array.min(axis=0), array.max(axis=0))])
        
    aminmax = np.vstack((arr.min(), arr.max()) for arr in tvalues)
    amins = aminmax[:, 0]
    aptps = aminmax.ptp(axis=1)

    
    ticks = np.vstack(tvalue[:n] for tvalue in tvalues)
    
    arrays = (array - amins) / aptps
    
    return arrays, ticks

#%%---------------------------------------------------------------------------#
def radar(titles, **title_opts):

    from matplotlib.path import Path
    from matplotlib.spines import Spine
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection


    def unit_poly_verts(theta):
        """Return vertices of polygon for subplot axes.
    
        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
        """
        x0, y0, r = [0.5] * 3
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts

    def draw_poly_patch(self):
        
        verts = unit_poly_verts(THETA)
        
        return plt.Polygon(verts, closed=True, edgecolor='k')
    
    TITLES = list(titles)
    
    THETA = np.linspace(0, 2*np.pi, len(TITLES), endpoint=False) + np.pi/2
    
    
    class RadarAxes(PolarAxes):
    
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            
            self.dim = len(TITLES)
            
            self.theta = np.linspace(0, 2*np.pi, self.dim+1) + np.pi/2
            
            super().__init__(*args, **kwargs)
                        
            super().grid(False)

            self.rgrid() 
            
            title_opts_ = dict(fontsize=12, weight="bold", color="black")
            
            title_opts_.update(title_opts)
            
            
            self.set_thetagrids(np.degrees(THETA).astype(np.float32), 
                                labels=TITLES, **title_opts_)
            
            self.set_xlim(0, np.pi * 2)
            self.set_ylim(0, 1)

            self.set_yticklabels([])
            for tick, d in zip(self.xaxis.get_ticklabels(), THETA):
                
                if np.pi / 2 < d < np.pi * 1.5:
                    tick.set_ha('right')
                elif np.pi * 1.5 < d:
                    tick.set_ha('left')

        def _gen_axes_patch(self):
            
            return draw_poly_patch(self)
        
    
        def _gen_axes_spines(self):

            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.
    
            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(THETA)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)
    
            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            
            return {'polar': spine}
            
        def rgrid(self, b=True, count=5, **kwargs):
            
            if hasattr(self, '_rgrids'):
                for col in self._rgrids:
                    
                    col.remove()
            
            if b:
            
                defaults = dict(color='grey', lw=0.5)       
                defaults.update(kwargs)
                
                dy = 1 / count
                            
                ys = np.ones_like(self.theta) * np.arange(dy, 1+dy, dy).reshape(-1, 1)
                
                xs = np.tile(self.theta, (count, 1))
                
                xys = np.dstack([xs, ys])
        
                line_segments = LineCollection(xys, **defaults)
                
                line_colls1 = self.add_collection(line_segments)
                
                xs = np.tile(THETA.reshape(-1, 1), 2)
                ys = np.tile([0, 1], (self.dim, 1))
                
                xys = np.dstack([xs, ys])
                line_segments = LineCollection(xys, **defaults)
                
                line_colls2 = self.add_collection(line_segments)
                                
                self._rgrids = (line_colls1, line_colls2)

        def set_zebra(self, **kwargs):
            
            assert hasattr(self, '_rgrids')
            
            defaults = dict(color='grey', alpha=0.2)
            
            defaults.update(kwargs)
            arr = np.dstack(p.vertices for p in self._rgrids[0].get_paths())
            self._rgrids[0].set_visible(defaults.pop('edge', False))
            xs, ys = arr[:, 0, :], arr[:, 1, :]
            
            xs = xs.T
            ys = ys.T
            

            return [self.fill_between(xs[2*i], ys[2*i], ys[2*i+1], 
                        zorder=0, **defaults) for i in range(len(xs) // 2)]
            
        def get_theta(self, title):
            
            return THETA[TITLES.index(title)]
        
        def scale(self, array, *arg,  **kwargs):
            
            assert hasattr(self, '_rgrids')

            origin = kwargs.pop('include_origin', False)
            apply = kwargs.pop('apply_tick', False)
            kind = kwargs.pop('kind', 'max')
            
            arr = np.dstack(p.vertices for p in self._rgrids[0].get_paths())

            _, _, n = arr.shape
            

            arrays, ticks = scale_radar(array, n=n+1, kind=kind)
            
            if not origin:
                
                ticks = ticks[:, 1:]
            
            if apply:
                
                for title, labels in zip(TITLES, ticks):
                    
                    self.set_rlabel(title, labels, include_origin=origin, **kwargs)
            
            return arrays, ticks
            
        def set_rlabel(self, title, labels, **kwargs):
            
            def get_txt():
                
                for xy, lbl in zip(xys, labels):
                    
                    yield self.text(*xy, fmt(lbl), **kwargs)
                    
            assert hasattr(self, '_rgrids')
            assert hasattr(labels, '__iter__')
            
            fmt = kwargs.pop('fmt', str)
            origin = kwargs.pop('include_origin', False)
            
            array = np.dstack(p.vertices for p in self._rgrids[0].get_paths())
            

            idx = TITLES.index(title)
            xys = array[idx].T
            if origin:

                xys = np.concatenate([[[0, 0]], xys], axis=0)            
            
            return list(get_txt())
        
        def plot(self, array, *args, **kwargs):
            
            clustered = kwargs.pop('clustered', False)            
            yarray = np.atleast_2d(np.asarray(array))
            
            assert np.all((yarray >= 0) & (yarray <= 1))
            num, dim = yarray.shape
            
            if dim != self.dim and num == self.dim:
                
                yarray = yarray.T
            
                num, dim = dim, num
            
            assert dim == self.dim
            
            yarray = np.concatenate([yarray, yarray[:, :1]], axis=1)
            
            if clustered:
                
                xys = np.dstack([np.tile(self.theta, (num, 1)), yarray])
                line_segments = LineCollection(xys, *args, **kwargs)
                
                lines = self.add_collection(line_segments)
            
            else:
                lines = super().plot(self.theta, yarray.T, *args, **kwargs)
            
            return lines
        
        def fill_lines(self, lines, **kwargs):
            
            colors = kwargs.pop('colors', [None]*len(lines))
            
            assert len(colors) == len(colors)
            
            for line, c in zip(lines, colors):
                
                color = line.get_color() if c is None else c
                super().fill(*line.get_xydata().T, color=color, **kwargs)
                
            return self
        
    register_projection(RadarAxes)
    
    return RadarAxes.name 

#%%---------------------------------------------------------------------------#

if __name__ == '__main__':
    
    
    fig = plt.figure(dpi=120)
    
    ax = fig.add_subplot(111, projection=radar(list('ABCDE')))
#    ax.rgrid(lw=1, linestyle='--',  alpha=0.4)
    
    lines = ax.plot([0.5, 0.4, 0.6, 0.8, 1])
    
    ax.fill_lines(lines, alpha=0.5, label='boo')
    ax.set_zebra( lw= 1)
    ax.set_rlabel('E', list('abcde'), color='b')
    ax.legend()
    plt.show()
    
    