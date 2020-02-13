import numpy as np


def keys(ob):
    try:
        return list(ob.keys())
    except:
        try:
            return ob.dtype.names
        except:
            raise Exception("Unable to get keys/names of obejct.")


def listify(list_or_scalar):
    try:
        list_or_scalar[0]
    except:
        return [list_or_scalar]
    else:
        return list_or_scalar

class Track:
    def __init__(self, initial_conds_record, data_record):
        self._inits = initial_conds_record
        self._data = data_record

    def __getattr__(self, attr):
        try:
            return self._inits[attr]
        except:
            pass

        try:
            return self._data[attr]
        except:
            raise AttributeError(f"Unknown attribute '{attr}' for this track.")


    def __getitem__(self, item):
        try:
            return self.__getattribute__(item)
        except:
            pass

        try:
            return self._data[item]
        except:
            pass

        try:
            return self._inits[item]
        except:
            raise KeyError(f"Unknown key '{item}' for this track")


    def __len__(self):
        """ Assumes all data records are the same length. """
        k = keys(self._data)[0]
        return len(self._data[k])

    @property
    def data(self):
        return self._data

    @property
    def inits(self):
        return self._inits



class TrackSet:
    def __init__(self, *tracks, ylabel='log_L', xlabel='log_Teff'):
        self._tracks = [*tracks]
        self._eps = [list() for _ in tracks] # Holds the output of ep funcs
        self._ep_funcs = []                  # Holds the ep funcs
        self._ylabel = ylabel
        self._xlabel = xlabel
        self._ep_labels = []

    def __iter__(self):
        return self._tracks.__iter__()

    def __getitem__(self, item):
        return self._tracks[item]

    def __getattr__(self, attr):
        if attr in self._ep_labels:
            ep_index = self._ep_labels.index(attr)
            return [self.get_ep_point(t, ep_index) for t in self.tracks_range]

        raise AttributeError(f"Invalid attribute: {attr}")

    def __len__(self):
        """ Get the number of tracks in this set """
        return len(self._tracks)

    def set_ylabel(self, label):
        """ Set the y axis label that ep points will be located with.
        All tracks must be able to return an array from calling `track.label`.
        """
        self._ylabel = label

    def set_xlabel(self, label):
        """ Set the x axis label that ep points will be located with.
        All tracks must be able to return an array from calling `track.label`.
        """
        self._xlabel = label


    def create_cols(self, func, label):
        """ Run a function on all current tracks to create a new column or multiple
        new columns on each track,
        and label it/them with the given `label(s)`.

        func: func(track) -> array or [array, ...]
        *len(track) and len(array) must be equal.
        """
        for t in self:
            cols = listify(func(t))
            label = listify(label)
            for c, l in zip(cols, label):
                t.__setattr__(l, c)


    def add_ep_func(self, func, label=None):
        """ Add a function to find ep (evolution points) points on tracks.
        When added, the function will be used on all current tracks,
        and all future tracks will be run by it aswell.
        func should return either a single index or a list of indexes.

        func: func(track) -> index(s) location in track.
        *the indexes can be floats, and the points will be interpolated.

        label: Optional label for the ep. Can be retrieved with self.ep_labels.
        If a label is provided, the n'th track's ep can also be
        retrieved with self.<label>[n].
        """
        for ep, t in zip(self._eps, self._tracks):
            ep += listify(func(t))

        self._ep_funcs.append(func)

        if label == None:
            label = f"EP {len(self._ep_labels) + 1}"
        self._ep_labels.append(label)

    def add_track(self, track):
        """ Add a track to the trackset. Any ep functions attached to this set
        will be run on the added track.

        Returns the index of the track added.
        """
        new_idx = len(self._tracks)
        self._tracks.append(track)
        eps = []
        for f in self._ep_funcs:
            eps += listify(f(track))

        self._eps.append(eps)
        return new_idx

    def get_ep_point(self, track_index, ep_index):
        """ Get a (X, Y) point for a track's ep point.

        track_index: the track number in this set.
        ep_index: the ep number for the track, or the label if it exists.
        """
        if ep_index in self._ep_labels:
            ep_index = self._ep_labels.index(ep_index)

        t = self._tracks[track_index]
        X = t[self.xlabel]
        Y = t[self.ylabel]
        ep = self._eps[track_index][ep_index]
        try:
            point = (X[ep], Y[ep])
        except:
            # interpolate instead
            interp_ax = np.arange(len(t))
            epi = int(ep)
            X_ep = np.interp(
                ep, interp_ax[epi:epi+2], X[epi:epi+2],
                left=X[epi], right=X[epi+1]
            )
            Y_ep = np.interp(
                ep, interp_ax[epi:epi+2], Y[epi:epi+2],
                left=Y[epi], right=Y[epi+1]
            )
            point = (X_ep, Y_ep)

        return np.array(point)

    def get_ep_points(self, track_index):
        """ Get a list of (log_T, log_L) points for a track.
        """
        # t = self._tracks[track_index]
        eps = self._eps[track_index]
        points = [self.get_ep_point(track_index, i) for i in range(len(eps))]
        return np.array(points)

    def interp_ep(self, track_index, ep_index, axis):
        """ Get the interpolated value at an ep point along the given axis.
        """
        ep = self._eps[track_index][ep_index]
        try:
            return axis[ep]
        except:
            # interpolate instead
            x_ax = np.arange(len(axis))
            epi = int(ep)
            p = np.interp(
                ep, x_ax[epi:epi+2], axis[epi:epi+2],
                left=axis[epi], right=axis[epi+1]
            )
            return p

    def delete_all_tracks(self):
        """ Delete all tracks and ep points, but keep ep functions and labels.
        """
        self._tracks = []
        self._eps = []
        # leave the funcs and labels

    # def delete_tracks(self, to_delete):
    #     """ Delete tracks with the given indexes. This will probably
    #     mess up any references to track indexes in this trackset.
    #     """

    @property
    def tracks_enum(self):
        return enumerate(self._tracks)

    @property
    def tracks_range(self):
        return range(len(self._tracks))

    @property
    def eps(self):
        "Evolutionary points for all tracks"
        return self._eps

    @property
    def num_eps(self):
        return len(self._ep_funcs)

    @property
    def ep_labels(self):
        return self._ep_labels


    def xlabel():
        doc = "The xlabel property."
        def fget(self):
            return self._xlabel
        def fset(self, value):
            self._xlabel = value
        def fdel(self):
            del self._xlabel
        return locals()
    xlabel = property(**xlabel())

    def ylabel():
        doc = "The ylabel property."
        def fget(self):
            return self._ylabel
        def fset(self, value):
            self._ylabel = value
        def fdel(self):
            del self._ylabel
        return locals()
    ylabel = property(**ylabel())
